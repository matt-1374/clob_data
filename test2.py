import time
import json
import csv
import os
import requests
import pytz
import sys
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
ASSET_SYMBOL = "XRPUSDT"
POLY_ASSET_SLUG = "xrp"

BINANCE_API_TICKER = "https://api.binance.com/api/v3/ticker/price"
BINANCE_API_KLINE = "https://api.binance.com/api/v3/klines"
BINANCE_API_TIME = "https://api.binance.com/api/v3/time"

POLY_GAMMA_API = "https://gamma-api.polymarket.com/events"
POLY_CLOB_MID_API = "https://clob.polymarket.com/midpoint"

POLL_INTERVAL = 1.0
TZ_ET = pytz.timezone("US/Eastern")

LOG_DIR = "data/xrp_1h"
LOG_ENABLED = True


class PolyTerminalMonitor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (PolyMonitor)"})

        self._binance_time_offset = timedelta(0)
        self._last_time_sync = 0.0

        self._log_fp = None
        self._log_writer = None
        self._log_path = None

    # ----------------------------
    # TIME SYNC
    # ----------------------------
    def sync_binance_time(self):
        try:
            r = self.session.get(BINANCE_API_TIME, timeout=2)
            if r.status_code == 200:
                server_ms = r.json()["serverTime"]
                server_dt = datetime.fromtimestamp(server_ms / 1000, tz=timezone.utc)
                local_dt = datetime.now(timezone.utc)
                self._binance_time_offset = server_dt - local_dt
                self._last_time_sync = time.monotonic()
        except Exception:
            # Silent fail; we just fall back to local UTC clock
            pass

    def now_utc(self):
        if time.monotonic() - self._last_time_sync > 60:
            self.sync_binance_time()
        return datetime.now(timezone.utc) + self._binance_time_offset

    # ----------------------------
    # DATA FETCHERS
    # ----------------------------
    def get_binance_price(self):
        try:
            r = self.session.get(
                BINANCE_API_TICKER, params={"symbol": ASSET_SYMBOL}, timeout=2
            )
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            pass
        return None

    def get_binance_hour_open(self, target_dt):
        # target_dt must be tz-aware
        target_dt = target_dt.astimezone(timezone.utc)
        ts_ms = int(
            target_dt.replace(minute=0, second=0, microsecond=0).timestamp() * 1000
        )

        try:
            r = self.session.get(
                BINANCE_API_KLINE,
                params={
                    "symbol": ASSET_SYMBOL,
                    "interval": "1h",
                    "startTime": ts_ms,
                    "limit": 1,
                },
                timeout=5,
            )
            if r.status_code == 200 and r.json():
                return float(r.json()[0][1])  # open
        except Exception:
            pass
        return None

    def get_midpoint_price(self, token_id):
        if token_id is None:
            return None
        try:
            r = self.session.get(
                POLY_CLOB_MID_API, params={"token_id": token_id}, timeout=2
            )
            if r.status_code == 200:
                data = r.json()
                mid = data.get("mid", None)
                if mid is None:
                    return None
                return float(mid)
        except Exception:
            pass
        return None

    # ----------------------------
    # LOGGING
    # ----------------------------
    def _ensure_logger(self, market_question, end_date_dt):
        if not LOG_ENABLED:
            return

        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = end_date_dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        path = os.path.join(LOG_DIR, f"xrp_updown_1h_{stamp}.csv")

        if self._log_path == path:
            return

        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass

        self._log_path = path
        new_file = not os.path.exists(path)

        self._log_fp = open(path, "a", newline="")
        self._log_writer = csv.writer(self._log_fp)

        if new_file:
            self._log_writer.writerow(
                ["ts_utc", "seconds_remaining", "xrp", "strike", "poly_up", "poly_down"]
            )
            self._log_fp.flush()

    def _log_row(self, ts_utc, seconds_remaining, xrp, strike, poly_up, poly_down):
        if not LOG_ENABLED or self._log_writer is None:
            return
        self._log_writer.writerow(
            [ts_utc.isoformat(), seconds_remaining, xrp, strike, poly_up, poly_down]
        )
        self._log_fp.flush()

    # ----------------------------
    # POLY DISCOVERY
    # ----------------------------
    def generate_slug(self, t):
        month = t.strftime("%B").lower()
        hour = t.strftime("%I").lstrip("0") or "0"
        ampm = t.strftime("%p").lower()
        return f"{POLY_ASSET_SLUG}-up-or-down-{month}-{t.day}-{hour}{ampm}-et"

    def find_active_market(self):
        now_et = datetime.now(TZ_ET)

        for i in range(2):
            slug = self.generate_slug(now_et + timedelta(hours=i))
            try:
                r = self.session.get(POLY_GAMMA_API, params={"slug": slug}, timeout=5)
                if r.status_code != 200:
                    continue
                payload = r.json()
                if not payload:
                    continue

                event = payload[0] if isinstance(payload, list) else payload
                markets = event.get("markets", [])

                for m in markets:
                    if m.get("closed"):
                        continue

                    raw_ids = m.get("clobTokenIds")
                    token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                    if not token_ids:
                        continue

                    yes_id = token_ids[0]
                    no_id = token_ids[1] if len(token_ids) > 1 else None

                    # sanity check YES exists
                    if self.get_midpoint_price(yes_id) is None:
                        continue

                    end_dt = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00"))
                    strike_dt = (end_dt - timedelta(hours=1)).replace(
                        minute=0, second=0, microsecond=0
                    )

                    strike = None
                    print(f"Found market: {m.get('question')}")
                    print(
                        f"Fetching strike (Binance 1h open) for {ASSET_SYMBOL} at {strike_dt.strftime('%Y-%m-%d %H:%M')} UTC..."
                    )

                    while strike is None:
                        if self.now_utc() > end_dt:
                            print("Market expired before strike was available.")
                            return None

                        strike = self.get_binance_hour_open(strike_dt)
                        if strike is None:
                            print("Waiting for Binance candle open... retrying in 10s")
                            time.sleep(10)

                    print(f"Strike confirmed: {strike:.6f}")
                    return {
                        "question": m.get("question", ""),
                        "end_date_dt": end_dt,
                        "strike": strike,
                        "yes_id": yes_id,
                        "no_id": no_id,
                    }
            except Exception:
                # silent; try next hour candidate
                pass

        return None

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    def run(self):
        self.sync_binance_time()
        print("Starting monitor... (Ctrl+C to stop)")

        while True:
            market = self.find_active_market()
            if not market:
                print("No active market found. Retrying in 10s...")
                time.sleep(10)
                continue

            self._ensure_logger(market["question"], market["end_date_dt"])
            print("Logging columns: ts_utc, seconds_remaining, xrp, strike, poly_up, poly_down")

            while True:
                loop_start = time.monotonic()
                now_utc = self.now_utc()

                if now_utc > market["end_date_dt"]:
                    print("\nMarket expired. Searching for next market...")
                    break

                xrp = self.get_binance_price()
                up = self.get_midpoint_price(market["yes_id"])
                down = self.get_midpoint_price(market["no_id"])

                # FIX 1: correct None-check (not truthiness)
                if (xrp is not None) and (up is not None) and (down is not None):
                    seconds_remaining = int(
                        (market["end_date_dt"] - now_utc).total_seconds()
                    )

                    self._log_row(
                        ts_utc=now_utc,
                        seconds_remaining=seconds_remaining,
                        xrp=xrp,
                        strike=market["strike"],
                        poly_up=up,
                        poly_down=down,
                    )

                    # FIX 2: live terminal output
                    print(
                        f"\r{now_utc.isoformat()} | left={seconds_remaining:4d}s | "
                        f"xrp={xrp:.6f} | strike={market['strike']:.6f} | up={up:.3f} | down={down:.3f}",
                        end="",
                        flush=True,
                    )

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        PolyTerminalMonitor().run()
    except KeyboardInterrupt:
        print("\nStopped.")
