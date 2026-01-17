import time
import json
import csv
import os
import requests
import pytz
import sys
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
ASSET_SYMBOL = "XRPUSDT"          # Binance spot symbol
POLY_ASSET_SLUG = "xrp"           # Polymarket slug prefix (e.g., xrp-up-or-down-...)

BINANCE_API_TICKER = "https://api.binance.com/api/v3/ticker/price"
BINANCE_API_KLINE = "https://api.binance.com/api/v3/klines"
BINANCE_API_TIME = "https://api.binance.com/api/v3/time"

POLY_GAMMA_API = "https://gamma-api.polymarket.com/events"
POLY_CLOB_MID_API = "https://clob.polymarket.com/midpoint"

POLL_INTERVAL = 1.0
TZ_ET = pytz.timezone("US/Eastern")

# Optional logging
LOG_DIR = "data/xrp_1h"
LOG_ENABLED = True

# ANSI Colors for Terminal
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
YELLOW = "\033[93m"


class PolyTerminalMonitor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; PolyTerminalMonitor/1.0)"}
        )

        self._binance_time_offset = timedelta(0)
        self._last_time_sync_mono = 0.0

        self._log_fp = None
        self._log_writer = None
        self._log_path = None

    # ----------------------------
    # TIME SYNC (Binance-anchored)
    # ----------------------------
    def sync_binance_time(self) -> None:
        try:
            resp = self.session.get(BINANCE_API_TIME, timeout=2)
            if resp.status_code == 200:
                server_ms = resp.json().get("serverTime")
                if server_ms is None:
                    return
                server_dt = datetime.fromtimestamp(server_ms / 1000.0, tz=timezone.utc)
                local_dt = datetime.now(timezone.utc)
                self._binance_time_offset = server_dt - local_dt
                self._last_time_sync_mono = time.monotonic()
        except Exception:
            pass

    def now_utc(self) -> datetime:
        if time.monotonic() - self._last_time_sync_mono > 60.0:
            self.sync_binance_time()
        return datetime.now(timezone.utc) + self._binance_time_offset

    # ----------------------------
    # DATA FETCHERS
    # ----------------------------
    def get_binance_price(self):
        try:
            resp = self.session.get(
                BINANCE_API_TICKER, params={"symbol": ASSET_SYMBOL}, timeout=2
            )
            if resp.status_code == 200:
                data = resp.json()
                return float(data["price"])
        except Exception:
            pass
        return None

    def get_binance_hour_open(self, target_dt: datetime):
        if target_dt.tzinfo is None:
            raise ValueError("target_dt must be timezone-aware")

        target_dt = target_dt.astimezone(timezone.utc)

        try:
            clean_dt = target_dt.replace(minute=0, second=0, microsecond=0)
            ts_ms = int(clean_dt.timestamp() * 1000)

            params = {
                "symbol": ASSET_SYMBOL,
                "interval": "1h",
                "startTime": ts_ms,
                "limit": 1,
            }

            resp = self.session.get(BINANCE_API_KLINE, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    return float(data[0][1])  # open
        except Exception as e:
            print(f" [!] Binance History Error: {e}")
        return None

    def get_midpoint_price(self, token_id):
        try:
            resp = self.session.get(
                POLY_CLOB_MID_API, params={"token_id": token_id}, timeout=2
            )
            if resp.status_code == 200:
                data = resp.json()
                mid = data.get("mid", None)
                if mid is None:
                    return None
                return float(mid)
        except Exception:
            pass
        return None

    # ----------------------------
    # LOGGING (ONLY requested fields)
    # ----------------------------
    def _ensure_logger(self, market_question: str, end_date_dt: datetime):
        if not LOG_ENABLED:
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        stamp = end_date_dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        safe_q = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (market_question or ""))[:80]
        path = os.path.join(LOG_DIR, f"{POLY_ASSET_SLUG}_updown_1h_{stamp}_{safe_q}.csv")

        if self._log_path == path and self._log_fp is not None:
            return

        try:
            if self._log_fp is not None:
                self._log_fp.close()
        except Exception:
            pass

        self._log_path = path
        new_file = not os.path.exists(path)

        self._log_fp = open(path, "a", newline="")
        self._log_writer = csv.writer(self._log_fp)

        if new_file:
            self._log_writer.writerow(["ts_utc", "xrp", "strike", "poly_up", "poly_down"])
            self._log_fp.flush()

    def _log_row(self, now_utc: datetime, xrp: float, strike: float, poly_up: float, poly_down: float):
        if not LOG_ENABLED or self._log_writer is None:
            return
        self._log_writer.writerow([now_utc.isoformat(), xrp, strike, poly_up, poly_down])
        self._log_fp.flush()

    # ----------------------------
    # POLY DISCOVERY (XRP 1H Up/Down)
    # ----------------------------
    def generate_slug(self, target_time: datetime) -> str:
        month = target_time.strftime("%B").lower()
        day = target_time.day
        hour_int = int(target_time.strftime("%I"))
        am_pm = target_time.strftime("%p").lower()
        hour_str = f"{hour_int}{am_pm}"
        return f"{POLY_ASSET_SLUG}-up-or-down-{month}-{day}-{hour_str}-et"

    def find_active_market(self):
        now_et = datetime.now(TZ_ET)

        print(f"{YELLOW}Scanning for active Polymarket events...{RESET}")

        for i in range(2):
            target_time = now_et + timedelta(hours=i)
            slug = self.generate_slug(target_time)

            try:
                resp = self.session.get(POLY_GAMMA_API, params={"slug": slug}, timeout=5)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                if not data:
                    continue

                event = data[0] if isinstance(data, list) else data
                markets = event.get("markets", [])

                for market in markets:
                    if market.get("closed"):
                        continue

                    raw_ids = market.get("clobTokenIds")
                    token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                    if not token_ids or len(token_ids) < 1:
                        continue

                    yes_token_id = token_ids[0]
                    no_token_id = token_ids[1] if len(token_ids) > 1 else None

                    # Sanity check: YES midpoint exists
                    yes_mid = self.get_midpoint_price(yes_token_id)
                    if yes_mid is None:
                        continue

                    end_date_str = (event.get("endDate", "") or "").replace("Z", "+00:00")
                    try:
                        end_date_dt = datetime.fromisoformat(end_date_str)
                    except Exception:
                        # Fallback format (older Gamma responses)
                        end_date_dt = (
                            datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S.000Z")
                            .replace(tzinfo=timezone.utc)
                        )

                    strike_time_dt = (end_date_dt - timedelta(hours=1)).replace(
                        minute=0, second=0, microsecond=0
                    )

                    strike = None
                    print(f"Found Market: {market.get('question')}")
                    print(
                        f"Fetching Strike (open) for {ASSET_SYMBOL} at {strike_time_dt.strftime('%Y-%m-%d %H:%M')} UTC..."
                    )

                    while strike is None:
                        if self.now_utc() > end_date_dt:
                            print(f"{RED}Market expired before strike found.{RESET}")
                            return None

                        strike = self.get_binance_hour_open(strike_time_dt)

                        if strike is None:
                            sys.stdout.write(
                                f"\r{YELLOW}Waiting for Binance candle open... (retrying in 10s){RESET}"
                            )
                            sys.stdout.flush()
                            time.sleep(10)
                        else:
                            print(f"\n{GREEN}Strike confirmed: {strike:.6f}{RESET}")

                    return {
                        "question": market.get("question"),
                        "strike": strike,
                        "end_date_dt": end_date_dt,
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                    }

            except Exception as e:
                print(f"{RED}Discovery Error: {e}{RESET}")
                time.sleep(2)

        return None

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    def run(self):
        self.sync_binance_time()

        while True:
            market = self.find_active_market()
            if not market:
                print(f"{RED}No active market found. Retrying in 10s...{RESET}")
                time.sleep(10)
                continue

            print("-" * 60)
            print(f"Tracking: {market['question']}")
            print(f"Logging: ts_utc, xrp, strike, poly_up, poly_down")
            print("-" * 60)

            self._ensure_logger(market["question"] or "xrp_updown_1h", market["end_date_dt"])

            while True:
                loop_start = time.monotonic()
                now_utc = self.now_utc()

                if now_utc > market["end_date_dt"]:
                    print(f"\n{YELLOW}Market expired. Finding next market...{RESET}")
                    break

                try:
                    xrp = self.get_binance_price()
                    up = self.get_midpoint_price(market["yes_token_id"])

                    if market["no_token_id"]:
                        down = self.get_midpoint_price(market["no_token_id"])
                    else:
                        down = (1.0 - up) if (up is not None) else None

                    if xrp is not None and up is not None and down is not None:
                        strike = market["strike"]
                        diff = xrp - strike

                        price_color = GREEN if xrp >= strike else RED
                        sign = "+" if diff > 0 else ""

                        output = (
                            f"XRP: {price_color}{xrp:.6f}{RESET} | "
                            f"Strike: {strike:.6f} ({sign}{diff:.6f}) | "
                            f"Poly UP: {up:.3f} | DOWN: {down:.3f}"
                        )
                        sys.stdout.write(f"\r{output}")
                        sys.stdout.flush()

                        self._log_row(now_utc=now_utc, xrp=xrp, strike=strike, poly_up=up, poly_down=down)

                except Exception as e:
                    sys.stdout.write(f"\nError: {e}\n")
                    sys.stdout.flush()

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        PolyTerminalMonitor().run()
    except KeyboardInterrupt:
        print("\n\nMonitor Stopped.")

