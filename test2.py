import time
import json
import csv
import os
import math
from collections import deque
import requests
import pytz
import sys
import numpy as np
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

# Warm-start from Binance klines (1m)
WARM_START_MINUTES = 300   # ~5 hours (buffer for 4h window)
WARM_START_SCALE = 12.0    # crude scale to map 1m ret^2 to ~5s RV scale

# ANSI Colors for Terminal
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
YELLOW = "\033[93m"

# ===============================
# REMAINING-RV LINEAR MODEL COEFFS
# ===============================
# NOTE: If your regression included an intercept, set it here.
INTERCEPT = 0.0

BETA_LOG_RV_15M = 0.512912
BETA_LOG_RV_1H = 0.293837
BETA_LOG_RV_4H = 0.120658
BETA_T_FRAC = -7.665737
BETA_SQRT_T_FRAC = 14.118690

EPS = 1e-12

# RV windows (5-second returns)
W_15M = 180   # 15m / 5s
W_1H = 720    # 1h / 5s
W_4H = 2880   # 4h / 5s


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predict_log_rv_remaining(
    log_rv_15m: float,
    log_rv_1h: float,
    log_rv_4h: float,
    time_left_frac: float,
) -> float:
    return (
        INTERCEPT
        + BETA_LOG_RV_15M * log_rv_15m
        + BETA_LOG_RV_1H * log_rv_1h
        + BETA_LOG_RV_4H * log_rv_4h
        + BETA_T_FRAC * time_left_frac
        + BETA_SQRT_T_FRAC * math.sqrt(max(time_left_frac, 0.0))
    )


def prob_close_above_strike(price_now: float, strike: float, log_rv_remaining_hat: float) -> float:
    """
    Assume remaining log return ~ N(0, RV_remaining).
    P(P_T >= strike) = 1 - Phi( log(strike/price_now) / sqrt(RV_rem) )
    """
    rv_rem = math.exp(log_rv_remaining_hat)
    sigma = math.sqrt(max(rv_rem, EPS))
    threshold = math.log(strike / price_now)
    z = threshold / sigma
    p = 1.0 - normal_cdf(z)
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


class RollingRVEstimator:
    """
    Builds 5-second log returns from live 1-second ticker snapshots and maintains trailing RV sums.

    Binning rule (matches offline dataset):
      - record a 5s "close" when epoch_second % 5 == 4
      - logret_5s uses consecutive 5s closes
      - rv windows are trailing sums of (logret_5s^2)

    NOTE:
      This uses Binance ticker snapshots, not true last-trade-per-second.
    """
    def __init__(self):
        self.last_5s_close_ts = None  # epoch seconds of last processed 5s close
        self.prev_5s_price = None

        self.ret2_15m = deque(maxlen=W_15M)
        self.ret2_1h = deque(maxlen=W_1H)
        self.ret2_4h = deque(maxlen=W_4H)

        self.sum_15m = 0.0
        self.sum_1h = 0.0
        self.sum_4h = 0.0

    def _push(self, dq: deque, current_sum: float, value: float) -> float:
        if len(dq) == dq.maxlen:
            current_sum -= dq[0]
        dq.append(value)
        current_sum += value
        return current_sum

    def update(self, now_utc: datetime, price: float) -> None:
        ts = int(now_utc.timestamp())
        if ts % 5 != 4:
            return
        if self.last_5s_close_ts == ts:
            return

        self.last_5s_close_ts = ts
        p = float(price)

        if self.prev_5s_price is None:
            self.prev_5s_price = p
            return

        if p <= 0.0 or self.prev_5s_price <= 0.0:
            self.prev_5s_price = p
            return

        logret = math.log(p / self.prev_5s_price)
        ret2 = logret * logret

        self.sum_15m = self._push(self.ret2_15m, self.sum_15m, ret2)
        self.sum_1h = self._push(self.ret2_1h, self.sum_1h, ret2)
        self.sum_4h = self._push(self.ret2_4h, self.sum_4h, ret2)

        self.prev_5s_price = p

    def get_log_rv_features(self):
        log_rv_15m = None
        log_rv_1h = None
        log_rv_4h = None

        if len(self.ret2_15m) == W_15M:
            log_rv_15m = math.log(max(self.sum_15m, EPS))
        if len(self.ret2_1h) == W_1H:
            log_rv_1h = math.log(max(self.sum_1h, EPS))
        if len(self.ret2_4h) == W_4H:
            log_rv_4h = math.log(max(self.sum_4h, EPS))

        return log_rv_15m, log_rv_1h, log_rv_4h

    def warmup_status(self) -> str:
        return f"5sRV warmup: 15m {len(self.ret2_15m)}/{W_15M}, 1h {len(self.ret2_1h)}/{W_1H}, 4h {len(self.ret2_4h)}/{W_4H}"

    def warm_reset(self) -> None:
        self.ret2_15m.clear()
        self.ret2_1h.clear()
        self.ret2_4h.clear()
        self.sum_15m = 0.0
        self.sum_1h = 0.0
        self.sum_4h = 0.0


class PolyTerminalMonitor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; PolyTerminalMonitor/1.0)"}
        )

        # Binance server-time alignment
        self._binance_time_offset = timedelta(0)
        self._last_time_sync_mono = 0.0

        # Logging
        self._log_fp = None
        self._log_writer = None
        self._log_path = None

        # Live RV estimator
        self._rv = RollingRVEstimator()

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
            pass

    def now_utc(self) -> datetime:
        """UTC time aligned to Binance server (best-effort)."""
        if time.monotonic() - self._last_time_sync_mono > 60.0:
            self.sync_binance_time()
        return datetime.now(timezone.utc) + self._binance_time_offset

    # ----------------------------
    # DATA FETCHERS
    # ----------------------------
    def get_binance_price(self):
        """Fetch live XRP price (spot)."""
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
        """Fetch the open price of the 1H candle starting at target_dt (timezone-aware, UTC)."""
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
                mid = data.get("mid", None)
                if mid is None:
                    return None
                return float(mid)
        except Exception:
            pass
        return None

    # ----------------------------
    # WARM START (NO ALWAYS-ON RV COLLECTOR)
    # ----------------------------
    def warm_start_rv_from_klines_1m(self, now_utc: datetime) -> None:
        """
        One-time warm start using Binance 1m klines over the last ~WARM_START_MINUTES.
        This avoids waiting ~4 hours for the 4h RV window to fill.

        This is an approximation because the model was trained on 5s features.
        We apply a crude scale factor (WARM_START_SCALE) to partially correct the RV scale.
        """
        end_ms = int(now_utc.timestamp() * 1000)
        start_ms = end_ms - WARM_START_MINUTES * 60_000

        params = {
            "symbol": ASSET_SYMBOL,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        try:
            resp = self.session.get(BINANCE_API_KLINE, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"{YELLOW}[WARN] warm start klines failed: HTTP {resp.status_code}{RESET}")
                return

            klines = resp.json()
            if not klines or len(klines) < 20:
                print(f"{YELLOW}[WARN] warm start klines empty/too short{RESET}")
                return

            closes = np.array([float(k[4]) for k in klines], dtype=np.float64)
            closes = closes[closes > 0]
            if closes.size < 20:
                print(f"{YELLOW}[WARN] warm start closes invalid{RESET}")
                return

            logp = np.log(closes)
            r = np.diff(logp)
            r2 = r * r

            # reset buffers
            self._rv.warm_reset()

            # Seed up to last 240 minutes = 4h at 1m resolution
            r2_tail = r2[-240:] if r2.size >= 240 else r2
            for v in r2_tail:
                vv = WARM_START_SCALE * float(v)
                self._rv.sum_4h = self._rv._push(self._rv.ret2_4h, self._rv.sum_4h, vv)
                self._rv.sum_1h = self._rv._push(self._rv.ret2_1h, self._rv.sum_1h, vv)
                self._rv.sum_15m = self._rv._push(self._rv.ret2_15m, self._rv.sum_15m, vv)

            print(f"{GREEN}[OK] warm start complete using {len(klines)}x 1m klines{RESET}")
            print(f"{GREEN}[OK] {self._rv.warmup_status()}{RESET}")

        except Exception as e:
            print(f"{YELLOW}[WARN] warm start exception: {e}{RESET}")
            return

    # ----------------------------
    # LOGGING
    # ----------------------------
    def _ensure_logger(self, market_question: str, end_date_dt: datetime):
        if not LOG_ENABLED:
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        stamp = end_date_dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        safe_q = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in market_question)[:80]
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
            self._log_writer.writerow(
                [
                    "ts_utc",
                    "price",
                    "strike",
                    "diff",
                    "yes_mid",
                    "no_mid",
                    "minutes_left",
                    "seconds_left",
                    "time_left_frac",
                    "log_rv_15m",
                    "log_rv_1h",
                    "log_rv_4h",
                    "log_rv_rem_hat",
                    "prob_up_model",
                ]
            )
            self._log_fp.flush()

    def _log_row(
        self,
        now_utc: datetime,
        price: float,
        strike: float,
        yes_mid: float,
        no_mid: float,
        end_date_dt: datetime,
        time_left_frac: float,
        log_rv_15m,
        log_rv_1h,
        log_rv_4h,
        log_rv_rem_hat,
        prob_up_model,
    ):
        if not LOG_ENABLED or self._log_writer is None:
            return
        time_left = end_date_dt - now_utc
        sec_left = max(0, int(time_left.total_seconds()))
        mm, ss = divmod(sec_left, 60)
        self._log_writer.writerow(
            [
                now_utc.isoformat(),
                price,
                strike,
                price - strike,
                yes_mid,
                no_mid,
                mm,
                ss,
                time_left_frac,
                log_rv_15m,
                log_rv_1h,
                log_rv_4h,
                log_rv_rem_hat,
                prob_up_model,
            ]
        )
        self._log_fp.flush()

    # ----------------------------
    # POLY DISCOVERY (XRP 1H Up/Down)
    # ----------------------------
    def generate_slug(self, target_time: datetime) -> str:
        month = target_time.strftime("%B").lower()
        day = target_time.day
        hour_int = int(target_time.strftime("%I"))  # 1..12
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

                    mid_price = self.get_midpoint_price(yes_token_id)
                    if mid_price is None:
                        continue

                    end_date_str = event.get("endDate", "").replace("Z", "+00:00")
                    try:
                        end_date_dt = datetime.fromisoformat(end_date_str)
                    except Exception:
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

        # One-time warm start (no always-on RV collector needed)
        now_utc = self.now_utc()
        self.warm_start_rv_from_klines_1m(now_utc)

        while True:
            market = self.find_active_market()
            if not market:
                print(f"{RED}No active market found. Retrying in 10s...{RESET}")
                time.sleep(10)
                continue

            print("-" * 60)
            print(f"Tracking: {market['question']}")
            print(f"Resolution basis: Binance {ASSET_SYMBOL} 1H open vs close")
            print(self._rv.warmup_status())
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
                    yes_price = self.get_midpoint_price(market["yes_token_id"])

                    if market["no_token_id"]:
                        no_price = self.get_midpoint_price(market["no_token_id"])
                    else:
                        no_price = (1.0 - yes_price) if (yes_price is not None) else None

                    if xrp is not None and yes_price is not None and no_price is not None:
                        # Update rolling estimator with live data
                        self._rv.update(now_utc, xrp)

                        time_left = market["end_date_dt"] - now_utc
                        sec_left = max(0, int(time_left.total_seconds()))
                        mm, ss = divmod(sec_left, 60)
                        time_left_frac = sec_left / 3600.0

                        strike = market["strike"]
                        diff = xrp - strike

                        log_rv_15m, log_rv_1h, log_rv_4h = self._rv.get_log_rv_features()

                        prob_up_model = None
                        log_rv_rem_hat = None
                        if (log_rv_15m is not None) and (log_rv_1h is not None) and (log_rv_4h is not None):
                            log_rv_rem_hat = predict_log_rv_remaining(
                                log_rv_15m=log_rv_15m,
                                log_rv_1h=log_rv_1h,
                                log_rv_4h=log_rv_4h,
                                time_left_frac=time_left_frac,
                            )
                            prob_up_model = prob_close_above_strike(
                                price_now=xrp,
                                strike=strike,
                                log_rv_remaining_hat=log_rv_rem_hat,
                            )

                        price_color = GREEN if xrp >= strike else RED
                        sign = "+" if diff > 0 else ""

                        if prob_up_model is None:
                            p_str = f"{YELLOW}warmup{RESET}"
                        else:
                            p_str = f"{prob_up_model:0.3f}"

                        output = (
                            f"Time: {mm:02}m {ss:02}s | "
                            f"XRP: {price_color}{xrp:.6f}{RESET} | "
                            f"Strike(O): {strike:.6f} ({sign}{diff:.6f}) | "
                            f"P(UP)_model: {p_str} | "
                            f"Poly UP(YES): {yes_price:.3f} | "
                            f"DOWN(NO): {no_price:.3f}"
                        )

                        sys.stdout.write(f"\r{output}")
                        sys.stdout.flush()

                        self._log_row(
                            now_utc=now_utc,
                            price=xrp,
                            strike=strike,
                            yes_mid=yes_price,
                            no_mid=no_price,
                            end_date_dt=market["end_date_dt"],
                            time_left_frac=time_left_frac,
                            log_rv_15m=log_rv_15m,
                            log_rv_1h=log_rv_1h,
                            log_rv_4h=log_rv_4h,
                            log_rv_rem_hat=log_rv_rem_hat,
                            prob_up_model=prob_up_model,
                        )

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
