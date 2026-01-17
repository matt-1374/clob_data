import time
import json
import requests
import pytz
import sys
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
BINANCE_API_TICKER = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_API_KLINE = "https://api.binance.com/api/v3/klines"
BINANCE_API_TIME = "https://api.binance.com/api/v3/time"

POLY_GAMMA_API = "https://gamma-api.polymarket.com/events"
POLY_CLOB_MID_API = "https://clob.polymarket.com/midpoint"

POLL_INTERVAL = 1.0
TZ_ET = pytz.timezone("US/Eastern")

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

        # Binance server-time alignment
        self._binance_time_offset = timedelta(0)
        self._last_time_sync_mono = 0.0

    # ----------------------------
    # TIME SYNC (Binance-anchored)
    # ----------------------------
    def sync_binance_time(self) -> None:
        """Sync local UTC clock to Binance server time (maintain an offset)."""
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
            # If sync fails, keep previous offset (or zero)
            pass

    def now_utc(self) -> datetime:
        """UTC time aligned to Binance server (best-effort)."""
        # re-sync every 60 seconds
        if time.monotonic() - self._last_time_sync_mono > 60.0:
            self.sync_binance_time()
        return datetime.now(timezone.utc) + self._binance_time_offset

    # ----------------------------
    # DATA FETCHERS
    # ----------------------------
    def get_binance_price(self):
        """Fetch live BTC price."""
        try:
            resp = self.session.get(BINANCE_API_TICKER, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return float(data["price"])
        except Exception:
            pass
        return None

    def get_binance_hour_open(self, target_dt: datetime):
        """Fetch the open price of the 1H candle starting at target_dt (timezone-aware, UTC)."""
        if target_dt.tzinfo is None:
            raise ValueError("target_dt must be timezone-aware")

        target_dt = target_dt.astimezone(timezone.utc)

        try:
            clean_dt = target_dt.replace(minute=0, second=0, microsecond=0)
            ts_ms = int(clean_dt.timestamp() * 1000)

            params = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "startTime": ts_ms,
                "limit": 1,
            }

            resp = self.session.get(BINANCE_API_KLINE, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    return float(data[0][1])  # Open Price
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
                return float(data.get("mid", 0))
        except Exception:
            pass
        return None

    # ----------------------------
    # POLY DISCOVERY
    # ----------------------------
    def generate_slug(self, target_time: datetime) -> str:
        month = target_time.strftime("%B").lower()
        day = target_time.day
        hour_int = int(target_time.strftime("%I"))
        am_pm = target_time.strftime("%p").lower()
        hour_str = f"{hour_int}{am_pm}"
        return f"bitcoin-up-or-down-{month}-{day}-{hour_str}-et"

    def find_active_market(self):
        now_et = datetime.now(TZ_ET)

        print(f"{YELLOW}Scanning for active Polymarket events...{RESET}")

        # Check Current Hour and Next Hour
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
                    yes_token_id = token_ids[0]
                    no_token_id = token_ids[1] if len(token_ids) > 1 else None

                    # Check validity via midpoint
                    mid_price = self.get_midpoint_price(yes_token_id)
                    if mid_price is None:
                        continue

                    # Parse Expiration (UTC)
                    end_date_str = event.get("endDate", "").replace("Z", "+00:00")
                    try:
                        end_date_dt = datetime.fromisoformat(end_date_str)
                    except Exception:
                        # fallback parser for older formats
                        end_date_dt = (
                            datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S.000Z")
                            .replace(tzinfo=timezone.utc)
                        )

                    # Calculate Strike Time (Top of previous hour, UTC)
                    strike_time_dt = (end_date_dt - timedelta(hours=1)).replace(
                        minute=0, second=0, microsecond=0
                    )

                    # Find Strike Price
                    strike = None
                    print(f"Found Market: {market.get('question')}")
                    print(
                        f"Fetching Strike Price for {strike_time_dt.strftime('%H:%M')} UTC..."
                    )

                    while strike is None:
                        if self.now_utc() > end_date_dt:
                            print(f"{RED}Market expired before Strike found.{RESET}")
                            return None

                        strike = self.get_binance_hour_open(strike_time_dt)

                        if strike is None:
                            sys.stdout.write(
                                f"\r{YELLOW}Waiting for Binance Candle Open... (Retrying in 10s){RESET}"
                            )
                            sys.stdout.flush()
                            time.sleep(10)
                        else:
                            print(f"\n{GREEN}Strike Price Confirmed: ${strike:,.2f}{RESET}")

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
        # initialize time sync early (best-effort)
        self.sync_binance_time()

        while True:
            # 1. FIND MARKET
            market = self.find_active_market()
            if not market:
                print(f"{RED}No active market found. Retrying in 10s...{RESET}")
                time.sleep(10)
                continue

            # 2. MONITOR LOOP
            print("-" * 60)
            print(f"Tracking: {market['question']}")
            print("-" * 60)

            while True:
                loop_start = time.monotonic()
                now_utc = self.now_utc()

                # Check Expiry
                if now_utc > market["end_date_dt"]:
                    print(f"\n{YELLOW}Market Expired. Finding next market...{RESET}")
                    break

                try:
                    # Fetch Data
                    btc = self.get_binance_price()
                    yes_price = self.get_midpoint_price(market["yes_token_id"])

                    # Logic to fetch "No" price if available, or calc inverse
                    if market["no_token_id"]:
                        no_price = self.get_midpoint_price(market["no_token_id"])
                    else:
                        no_price = 1.0 - yes_price if yes_price else 0

                    if btc is not None and yes_price is not None:
                        # Time Calc
                        time_left = market["end_date_dt"] - now_utc
                        mm, ss = divmod(int(time_left.total_seconds()), 60)

                        # Formatting
                        strike = market["strike"]
                        diff = btc - strike

                        # Color logic based on winning condition
                        price_color = GREEN if btc > strike else RED
                        sign = "+" if diff > 0 else ""

                        # OUTPUT LINE
                        output = (
                            f"Time: {mm:02}m {ss:02}s | "
                            f"BTC: {price_color}${btc:,.2f}{RESET} | "
                            f"Strike: ${strike:,.2f} ({sign}${diff:.2f}) | "
                            f"Poly YES: {yes_price:.3f} | "
                            f"NO: {no_price:.3f}"
                        )

                        # Print with Carriage Return (\r) to stay on same line
                        sys.stdout.write(f"\r{output}")
                        sys.stdout.flush()

                except Exception as e:
                    # Don't crash on temporary net errors, just print on new line
                    sys.stdout.write(f"\nError: {e}\n")
                    sys.stdout.flush()

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        PolyTerminalMonitor().run()
    except KeyboardInterrupt:
        print("\n\nMonitor Stopped.")
