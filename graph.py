import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(df: pd.DataFrame, preds: dict, asset="btc", steps=295):
    """
    df: original training DataFrame (with 'ds_naive_utc' and asset column)
    preds: forecast dict returned by train_and_forecast()
    asset: which asset to plot
    """
    # --- prepare training series ---
    hist = df[["ds_naive_utc", asset]].dropna().rename(columns={"ds_naive_utc": "ds", asset: "y"})
    hist = hist.sort_values("ds").tail(288 * 3)  # last 3 days for clarity

    # --- prepare forecast series ---
    fc_rows = []
    for ts, cols in preds.items():
        if asset in cols:
            fc_rows.append({"ds": pd.Timestamp(ts), "yhat": cols[asset]})
    fc_df = pd.DataFrame(fc_rows).sort_values("ds")

    # --- plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(hist["ds"], hist["y"], label="Historical", color="steelblue")
    plt.plot(fc_df["ds"], fc_df["yhat"], label="Forecast", color="orange")

    # mark the forecast start
    if not hist.empty and not fc_df.empty:
        plt.axvline(hist["ds"].max(), color="gray", linestyle="--", alpha=0.7)

    plt.title(f"{asset.upper()} — ARIMA Forecast ({steps}×5min steps ≈ {steps*5/60:.1f}h)")
    plt.xlabel("UTC time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
