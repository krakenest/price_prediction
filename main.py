# price_sampler_offset_hist_1m.py (UTC-safe, LSTM + XGBoost hybrid)
import os, sys, time, requests, datetime as dt, warnings, math
from dotenv import load_dotenv
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from graph import plot_forecast
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ==== Config / API ====
# Externalized secrets/config: set env vars CMC_API_KEY and DATABASE_URL
load_dotenv()
API_KEY = os.getenv("CMC_API_KEY", "")
DB_URL  = os.getenv("DATABASE_URL", "")
TABLE = "sn50_price"

# Networking and training constants
DEFAULT_FETCH_INTERVAL = "5m"
REQUEST_TIMEOUT_S = 45
MAX_RETRY_ATTEMPTS = 6
INITIAL_BACKOFF_S = 1.5
BACKOFF_MULTIPLIER = 1.8
CHUNK_DAYS = 7

DEFAULT_HORIZON_STEPS = 295
DEFAULT_LOOKBACK = 120
DEFAULT_EPOCHS = 40
DEFAULT_LSTM_HIDDEN = 96
DEFAULT_LSTM_LAYERS = 2
DEFAULT_LR = 1e-3

CMC_HIST = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"

if not API_KEY:
    warnings.warn("CMC_API_KEY env var not set; API requests may fail.")
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var not set. Please export DATABASE_URL before running.")

ENGINE = create_engine(DB_URL, pool_pre_ping=True)

SESSION = requests.Session()
SESSION.headers.update({
    "X-CMC_PRO_API_KEY": API_KEY,
    "Accept": "application/json",
    "User-Agent": "cmc-price-5m-utc/1.0",
})

# ---- ID map ----
ID_MAP = {
    "1": "btc",
    "1027": "eth",
    "5176": "xaut",
    "5426": "sol",
}
ID_LIST = list(ID_MAP.keys())

# ---- UTC helpers ----
UTC = dt.timezone.utc
def ensure_aware_utc(t: dt.datetime) -> dt.datetime:
    """Return `t` as timezone-aware UTC.

    Naive datetimes are assumed to be UTC.
    """
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)

def iso_z(x: dt.datetime) -> str:
    """Format datetime as ISO8601 with trailing 'Z' and zeroed seconds."""
    x = ensure_aware_utc(x).replace(second=0, microsecond=0)
    return x.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

def floor_5m(t: dt.datetime) -> dt.datetime:
    """Floor a datetime to the previous 5-minute boundary in UTC."""
    t = ensure_aware_utc(t).replace(second=0, microsecond=0)
    return t - dt.timedelta(minutes=t.minute % 5)

# ---- Weekly iterator (7-day windows, UTC) ----
def iter_week_slices(start_utc: dt.datetime, end_utc: dt.datetime):
    """Yield [start, end) 7-day slices between two UTC datetimes."""
    cur = ensure_aware_utc(start_utc)
    end_utc = ensure_aware_utc(end_utc)
    step = dt.timedelta(days=CHUNK_DAYS)
    while cur < end_utc:
        nxt = min(cur + step, end_utc)
        yield (cur, nxt)
        cur = nxt

# ---- DB ----
def get_last_trained_ts():
    """Return most recent `ts_utc` marked as trained, or None."""
    sql = text(f"SELECT ts_utc FROM {TABLE} WHERE is_trained=true ORDER BY ts_utc DESC LIMIT 1")
    with ENGINE.begin() as con:
        row = con.execute(sql).fetchone()
        if not row: return None
        ts = row[0]
        return ensure_aware_utc(ts) if isinstance(ts, dt.datetime) else None

def upsert_rows(rows: dict[dt.datetime, dict[str, float]], trained: bool = True):
    """Upsert aggregated rows into `{TABLE}`; respects `is_trained` lock semantics."""
    if not rows: return
    sql = text(f"""
      INSERT INTO {TABLE} (ts_utc, btc, eth, xaut, sol, is_trained, updated_at)
      VALUES (:ts, :btc, :eth, :xaut, :sol, :trained, now() AT TIME ZONE 'utc')
      ON CONFLICT (ts_utc) DO UPDATE
      SET
        btc  = CASE WHEN {TABLE}.is_trained THEN {TABLE}.btc
                    ELSE COALESCE(EXCLUDED.btc,  {TABLE}.btc) END,
        eth  = CASE WHEN {TABLE}.is_trained THEN {TABLE}.eth
                    ELSE COALESCE(EXCLUDED.eth,  {TABLE}.eth) END,
        xaut = CASE WHEN {TABLE}.is_trained THEN {TABLE}.xaut
                    ELSE COALESCE(EXCLUDED.xaut, {TABLE}.xaut) END,
        sol  = CASE WHEN {TABLE}.is_trained THEN {TABLE}.sol
                    ELSE COALESCE(EXCLUDED.sol,  {TABLE}.sol) END,
        is_trained = {TABLE}.is_trained OR EXCLUDED.is_trained,
        updated_at = now() AT TIME ZONE 'utc'
      WHERE
        NOT {TABLE}.is_trained
        AND (
          COALESCE(EXCLUDED.btc,  {TABLE}.btc)  IS DISTINCT FROM {TABLE}.btc OR
          COALESCE(EXCLUDED.eth,  {TABLE}.eth)  IS DISTINCT FROM {TABLE}.eth OR
          COALESCE(EXCLUDED.xaut, {TABLE}.xaut) IS DISTINCT FROM {TABLE}.xaut OR
          COALESCE(EXCLUDED.sol,  {TABLE}.sol)  IS DISTINCT FROM {TABLE}.sol  OR
          ({TABLE}.is_trained OR EXCLUDED.is_trained) IS DISTINCT FROM {TABLE}.is_trained
        );
    """)
    payload = [{
        "ts": ensure_aware_utc(ts),
        "btc": cols.get("btc"),
        "eth": cols.get("eth"),
        "xaut": cols.get("xaut"),
        "sol": cols.get("sol"),
        "trained": trained,
    } for ts, cols in rows.items()]
    with ENGINE.begin() as con:
        con.execute(sql, payload)

# ---- Fetcher (proper 7-day chunking + retries) ----
def fetch_quotes_5m_chunked(start_utc: dt.datetime, end_utc: dt.datetime, convert: str = "USD"):
    """Stream CMC quotes in CHUNK_DAYS windows with retry/backoff."""
    WINDOW_DAYS = CHUNK_DAYS
    cursor_start = ensure_aware_utc(start_utc)
    end_utc = ensure_aware_utc(end_utc)
    while cursor_start < end_utc:
        cursor_end = min(cursor_start + dt.timedelta(days=WINDOW_DAYS), end_utc)
        params = {
            "id": ",".join(ID_LIST),
            "time_start": iso_z(cursor_start),
            "time_end": iso_z(cursor_end),
            "interval": DEFAULT_FETCH_INTERVAL,
            "convert": convert,
        }
        backoff = INITIAL_BACKOFF_S
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                r = SESSION.get(CMC_HIST, params=params, timeout=REQUEST_TIMEOUT_S)
                if r.status_code == 429:
                    time.sleep(backoff); backoff *= BACKOFF_MULTIPLIER; continue
                r.raise_for_status()
                payload = r.json() or {}
                yield payload.get("data", {})
                break
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(backoff); backoff *= BACKOFF_MULTIPLIER; continue
            except requests.HTTPError:
                time.sleep(backoff); backoff *= BACKOFF_MULTIPLIER; continue
        cursor_start = cursor_end

# ---- Main fetch ----
def main_fetch():
    """Fetch new 5m prices from CMC and upsert into the database."""
    now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
    last = get_last_trained_ts()
    start = ensure_aware_utc(last if last else now - dt.timedelta(days=28))
    end = ensure_aware_utc(now)
    if start >= end:
        print("Nothing new to fetch."); return

    total_rows = 0
    def maybe_record(ts_iso: str, key: str, price: float, buckets: dict):
        if ts_iso is None or price is None: return
        ts = dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        row_ts = floor_5m(ts)
        buckets[row_ts][key] = price

    for s, e in iter_week_slices(start, end):
        week_rows = 0
        buckets: dict[dt.datetime, dict[str, float]] = defaultdict(dict)
        for data_block in fetch_quotes_5m_chunked(s, e, convert="USD"):
            for cid, node in (data_block or {}).items():
                key = ID_MAP.get(str(cid))
                if not key: continue
                for q in node.get("quotes", []):
                    ts_iso = q.get("timestamp")
                    px = q.get("quote", {}).get("USD", {}).get("price")
                    maybe_record(ts_iso, key, px, buckets)
            if len(buckets) >= 5000:
                upsert_rows(buckets)
                n = len(buckets); week_rows += n; total_rows += n; buckets.clear()
        if buckets:
            upsert_rows(buckets)
            n = len(buckets); week_rows += n; total_rows += n; buckets.clear()
        print(f"{TABLE}: week slice upserted {week_rows} rows [{s.isoformat()} .. {e.isoformat()})")

    print(f"{TABLE}: total ~{total_rows} rows from {start.isoformat()} to {end.isoformat()}")

# ---- Training data ----
def load_training_data(day_back: int = 6):
    """Load recent trained rows and return a DataFrame with naive UTC timestamp column."""
    since = dt.datetime.now(UTC) - dt.timedelta(days=day_back)
    sql = text(f"""
        SELECT ts_utc, btc, eth, xaut, sol
        FROM {TABLE}
        WHERE is_trained = true AND ts_utc >= :since
        ORDER BY ts_utc ASC
    """)
    with ENGINE.begin() as con:
        df = pd.read_sql(sql, con, params={"since": since})
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ds_naive_utc"] = df["ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df

def to_5min_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 5-minute frequency, forward-filled, on a naive datetime column `ds`."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"], utc=False)
    df["ds"] = df["ds"].dt.tz_localize(None).dt.floor("5min")
    df = df.drop_duplicates("ds").sort_values("ds")
    start, end = df["ds"].min(), df["ds"].max()
    idx = pd.date_range(start, end, freq="5min")
    df = df.set_index("ds").reindex(idx).ffill().reset_index().rename(columns={"index":"ds"})
    return df

# ====== LSTM + XGBoost hybrid ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden=64, layers=2, output_dim=295, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden, output_dim)
    def forward(self, x):              # x: (B, T, 1)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # (B, H)
        return self.fc(last)           # (B, 295)

def _technical_features(window_prices: np.ndarray) -> dict:
    """Compute simple technical features from a price window."""
    window_prices = np.asarray(window_prices, dtype=float)
    last_price = float(window_prices[-1])
    window_prices = np.clip(np.nan_to_num(window_prices, nan=last_price, posinf=last_price, neginf=last_price), 1e-9, None)

    ret = np.diff(np.log(window_prices))
    if ret.size == 0:
        ret = np.array([0.0], dtype=float)

    vol = float(np.nanstd(ret)) or 0.0
    mean = float(np.nanmean(ret)) if np.isfinite(np.nanmean(ret)) else 0.0

    n = 14
    gains = np.clip(ret, 0, None)
    losses = -np.clip(ret, None, 0)
    g = float(np.nanmean(gains[-n:])) if len(ret) >= n else float(np.nanmean(gains))
    l = float(np.nanmean(losses[-n:])) if len(ret) >= n else float(np.nanmean(losses))
    rs = (g + 1e-9) / (l + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return {
        "last_price": last_price,
        "ret_mean": mean,
        "ret_vol": vol,
        "rsi14": float(rsi),
    }


def _make_sequences(prices: np.ndarray, horizon: int, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    # model on log prices for stability
    lp = np.log(np.maximum(prices, 1e-9))
    X, y = [], []
    for i in range(lookback, len(lp) - horizon + 1):
        X.append(lp[i - lookback:i])           # (lookback,)
        y.append(lp[i:i + horizon])            # (horizon,)
    return np.array(X), np.array(y)

def train_lstm(
    prices: np.ndarray,
    horizon: int = DEFAULT_HORIZON_STEPS,
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = DEFAULT_EPOCHS,
    hidden: int = DEFAULT_LSTM_HIDDEN,
    layers: int = DEFAULT_LSTM_LAYERS,
    lr: float = DEFAULT_LR,
) -> Tuple[Seq2SeqLSTM, StandardScaler]:
    """Train a simple seq2seq LSTM on log-price windows and return model and scaler."""
    # Scale log prices; model works better with normalized inputs
    lp = np.log(np.maximum(prices, 1e-9)).reshape(-1, 1)
    scaler = StandardScaler()
    lp_scaled = scaler.fit_transform(lp).ravel()

    Xlp, Ylp = _make_sequences(np.exp(lp_scaled), horizon, lookback)  # trick: scale via exp to keep >0
    
    # Shapes
    X_t = torch.tensor(Xlp[:, :, None], dtype=torch.float32)
    Y_t = torch.tensor(Ylp, dtype=torch.float32)

    # Simple split
    idx_train, idx_val = train_test_split(np.arange(len(X_t)), test_size=0.1, shuffle=False)
    Xtr, Ytr = X_t[idx_train], Y_t[idx_train]
    Xva, Yva = X_t[idx_val], Y_t[idx_val]

    model = Seq2SeqLSTM(input_dim=1, hidden=hidden, layers=layers, output_dim=horizon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xtr.to(DEVICE))
        loss = crit(pred, Ytr.to(DEVICE))
        loss.backward()
        opt.step()
        if (ep+1) % 10 == 0 and len(Xva) > 0:
            model.eval()
            with torch.no_grad():
                vpred = model(Xva.to(DEVICE))
                vloss = crit(vpred, Yva.to(DEVICE)).item()
            model.train()
            print(f"[LSTM] epoch {ep+1}/{epochs} train_loss={loss.item():.5f} val_loss={vloss:.5f}")
    return model, scaler

def lstm_predict_prices(
    model: Seq2SeqLSTM,
    scaler: StandardScaler,
    history_prices: np.ndarray,
    horizon: int = DEFAULT_HORIZON_STEPS,
    lookback: int = DEFAULT_LOOKBACK,
) -> np.ndarray:
    """Predict future prices using the trained LSTM given a lookback window."""
    assert len(history_prices) >= lookback, "Not enough history for LSTM input"

    # ensure history has no NaN/inf
    history_prices = np.asarray(history_prices, dtype=float)
    last_valid = float(history_prices[np.isfinite(history_prices)][-1])
    history_prices = np.nan_to_num(history_prices, nan=last_valid, posinf=last_valid, neginf=last_valid)
    history_prices = np.clip(history_prices, 1e-9, None)

    lp = np.log(history_prices).reshape(-1, 1)
    lp_scaled = scaler.transform(lp).ravel()

    xseq = np.exp(lp_scaled[-lookback:])
    xseq = np.log(xseq).astype(np.float32)[None, :, None]

    model.eval()
    with torch.no_grad():
        yhat_log_scaled = model(torch.tensor(xseq, dtype=torch.float32, device=DEVICE)).cpu().numpy()[0]

    # Undo scaling trick (stay numerically safe)
    yhat_scaled = np.exp(yhat_log_scaled)
    yhat_log = np.log(np.clip(yhat_scaled, 1e-12, None))

    mean = float(np.ravel(scaler.mean_)[0])
    if hasattr(scaler, "scale_"):
        scale = float(np.ravel(scaler.scale_)[0])
    elif hasattr(scaler, "var_"):
        scale = float(np.sqrt(np.ravel(scaler.var_)[0]))
    else:
        scale = 1.0
    yhat_log_orig = yhat_log * scale + mean


    prices_pred = np.exp(np.clip(yhat_log_orig, np.log(1e-9), np.log(1e12)))

    # Final guard: replace any non-finite with last_valid price
    prices_pred = np.nan_to_num(prices_pred, nan=last_valid, posinf=last_valid, neginf=last_valid)
    prices_pred = np.clip(prices_pred, 1e-9, last_valid * 10.0)

    return prices_pred.astype(float)


def build_xgb_training(
    prices: np.ndarray,
    lstm_model: Seq2SeqLSTM,
    scaler: StandardScaler,
    horizon: int = DEFAULT_HORIZON_STEPS,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct feature matrix and targets for XGBoost residual correction."""
    prices = np.asarray(prices, dtype=float)

    # --- Guard: if everything is non-finite, return empty
    finite_mask = np.isfinite(prices)
    if not finite_mask.any():
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

    last_valid = float(prices[finite_mask][-1])
    prices = np.nan_to_num(prices, nan=last_valid, posinf=last_valid, neginf=last_valid)
    prices = np.clip(prices, 1e-9, None)

    # --- Guard: enough length?
    if len(prices) < (lookback + horizon + 1):
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

    feats, tgts = [], []
    for i in range(lookback, len(prices) - horizon):
        window = prices[i - lookback:i]
        if not np.all(np.isfinite(window)):
            continue

        true_future = prices[i:i + horizon]  # (H,)
        lstm_future = lstm_predict_prices(lstm_model, scaler, window, horizon, lookback)

        # Skip bad samples
        if (not np.all(np.isfinite(lstm_future))) or (not np.all(np.isfinite(true_future))):
            continue

        tech = _technical_features(window)

        # Precompute for speed
        lp = float(tech["last_price"])
        rm = float(tech["ret_mean"])
        rv = float(tech["ret_vol"])
        rsi = float(tech["rsi14"])

        for h in range(horizon):
            step_norm = h / max(1, (horizon - 1))
            f = [
                float(lstm_future[h]),
                lp, rm, rv, rsi,
                float(step_norm),
            ]
            y = float(true_future[h] - lstm_future[h])  # residual (price space)

            # Filter here as well
            if not (np.all(np.isfinite(f)) and np.isfinite(y)):
                continue

            feats.append(f)
            tgts.append(y)

    if len(feats) == 0:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(tgts, dtype=np.float32)

    # --- Final sanitation
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[m], y[m]

    # --- Winsorize residuals to avoid "value too large"
    if y.size:
        med = np.median(y)
        mad = np.median(np.abs(y - med)) + 1e-9
        cap = float(med + 8.0 * 1.4826 * mad)  # ~8σ equivalent
        floor = float(med - 8.0 * 1.4826 * mad)
        y = np.clip(y, floor, cap)

    return X, y


def train_xgb(feats: np.ndarray, tgts: np.ndarray) -> XGBRegressor:
    """Train an XGBoost regressor; handle tiny datasets gracefully."""
    # Empty or almost-empty → trivial model
    if feats.size == 0 or tgts.size == 0:
        dummy = XGBRegressor(n_estimators=1, tree_method="hist", random_state=42)
        Xd = np.zeros((1, 6), dtype=np.float32)
        yd = np.array([0.0], dtype=np.float32)
        dummy.fit(Xd, yd)
        return dummy

    # Finite check
    mask = np.isfinite(feats).all(axis=1) & np.isfinite(tgts)
    feats = feats[mask].astype(np.float32, copy=False)
    tgts = tgts[mask].astype(np.float32, copy=False)

    # Still too few? Fit a tiny model (no eval_set)
    if len(tgts) < 30:
        small = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1e-3,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=0,
        )
        small.fit(feats, tgts)
        return small

    # Time-ordered split (no shuffle). Ensure non-empty val.
    val_size = max(1, int(0.1 * len(tgts)))
    xtr, xva = feats[:-val_size], feats[-val_size:]
    ytr, yva = tgts[:-val_size], tgts[-val_size:]

    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1e-3,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=0,
        objective="reg:squarederror",
        eval_metric="rmse",
    )

    if len(yva) > 0:
        xgb.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    else:
        xgb.fit(xtr, ytr, verbose=False)

    return xgb

def hybrid_forecast(
    prices: np.ndarray,
    lstm_model: Seq2SeqLSTM,
    scaler: StandardScaler,
    xgb: XGBRegressor,
    horizon: int = DEFAULT_HORIZON_STEPS,
    lookback: int = DEFAULT_LOOKBACK,
) -> np.ndarray:
    """Combine LSTM baseline with XGB-predicted residuals to produce final forecast."""
    window = prices[-lookback:]
    lstm_future = lstm_predict_prices(lstm_model, scaler, window, horizon, lookback)
    tech = _technical_features(window)
    feats = np.column_stack([
        lstm_future,
        np.full(horizon, tech["last_price"]),
        np.full(horizon, tech["ret_mean"]),
        np.full(horizon, tech["ret_vol"]),
        np.full(horizon, tech["rsi14"]),
        np.linspace(0.0, 1.0, horizon)
    ])
    residuals = xgb.predict(feats)
    return (lstm_future + residuals).astype(float)

# ====== Orchestration over assets ======
def train_and_forecast(df: pd.DataFrame, steps: int = DEFAULT_HORIZON_STEPS, lookback: int = DEFAULT_LOOKBACK):
    """
    df must have: ['ds_naive_utc', 'btc','eth','xaut','sol'].
    Returns dict[UTC_ts][asset] = prediction for exactly `steps` future 5-min points.
    """
    forecasts = defaultdict(dict)
    assets = ["btc", "eth", "xaut", "sol"]

    for asset in assets:
        sub = df[["ds_naive_utc", asset]].dropna()
        if sub.empty:
            continue
        sub = sub.rename(columns={"ds_naive_utc": "ds", asset: "y"}).sort_values("ds")
        sub = to_5min_naive(sub)
        y = sub["y"].astype(float).to_numpy()

        if len(y) < (lookback + steps + 50):
            print(f"[{asset}] Not enough data for LSTM+XGB (len={len(y)}). Skipping.")
            continue

        # ---- Train LSTM on full history
        lstm_model, scaler = train_lstm(
            y,
            horizon=steps,
            lookback=lookback,
            epochs=DEFAULT_EPOCHS,
            hidden=DEFAULT_LSTM_HIDDEN,
            layers=DEFAULT_LSTM_LAYERS,
            lr=DEFAULT_LR,
        )

        # ---- Train XGB residual corrector
        feats, tgts = build_xgb_training(y, lstm_model, scaler, horizon=steps, lookback=lookback)
        xgb = train_xgb(feats, tgts)

        # ---- Forecast steps ahead
        preds = hybrid_forecast(y, lstm_model, scaler, xgb, horizon=steps, lookback=lookback)

        last_train = sub["ds"].max()  # tz-naive
        future_idx = pd.date_range(start=last_train + pd.Timedelta(minutes=5), periods=steps, freq="5min")
        for ts, val in zip(future_idx, preds):
            ts_utc = pd.Timestamp(ts, tz="UTC")
            forecasts[ts_utc][asset] = float(val)

    return forecasts

# ====== Train entrypoint ======
def main_train():
    """Load data, train hybrid models per asset, write forecasts, and plot."""
    df = load_training_data(day_back=30)
    if df.empty:
        print("No training data found."); return
    steps = DEFAULT_HORIZON_STEPS
    preds = train_and_forecast(df, steps=steps, lookback=DEFAULT_LOOKBACK)

    upsert_rows(preds, trained=False)
    # Plot each asset (your existing plotting util)
    for a in ["btc", "eth", "xaut", "sol"]:
        try:
            plot_forecast(df, preds, asset=a)
        except Exception as e:
            print(f"plot_forecast failed for {a}: {e}")
            
    print(f"{TABLE}: prepared {len(preds)} 5m forecast timestamps (is_trained=false)")

# ====== CLI ======
if __name__ == "__main__":
    start_all = time.perf_counter()
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] Starting main_fetch()...")
    t0 = time.perf_counter(); main_fetch(); fetch_time = time.perf_counter() - t0
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] main_fetch() completed in {fetch_time/60:.2f} min")

    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] Starting main_train()...")
    t1 = time.perf_counter(); main_train(); train_time = time.perf_counter() - t1
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] main_train() completed in {train_time/60:.2f} min")

    total = time.perf_counter() - start_all
    print(f"✅ Total runtime: {total/60:.2f} min ({total:.1f} sec)")
