import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

sent = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data.csv")

sent["date"] = pd.to_datetime(sent["date"])
sent = sent.rename(columns={"classification": "sentiment", "value": "sentiment_value"})
sent = sent[["date", "sentiment_value", "sentiment"]].drop_duplicates(subset="date")

# Timestamp (epoch ms) got mangled on export - rounds thousands of rows to the same value.
# Timestamp IST is a plain string so it didn't lose precision, use that and shift to UTC.
trades["datetime_ist"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M")
trades["datetime_utc"] = trades["datetime_ist"] - pd.Timedelta(hours=5, minutes=30)
trades["date"] = trades["datetime_utc"].dt.normalize()

trades = trades.rename(columns={
    "Account": "account", "Coin": "coin", "Execution Price": "price",
    "Size Tokens": "size_tokens", "Size USD": "size_usd", "Side": "side",
    "Start Position": "start_position", "Direction": "direction",
    "Closed PnL": "closed_pnl", "Fee": "fee", "Crossed": "crossed",
    "Order ID": "order_id", "Trade ID": "trade_id"
})

keep_cols = ["account", "coin", "price", "size_tokens", "size_usd", "side",
             "direction", "start_position", "closed_pnl", "fee", "crossed",
             "order_id", "trade_id", "datetime_ist", "datetime_utc", "date"]
trades = trades[keep_cols]

print("trades date range:", trades["date"].min(), trades["date"].max())
print("sentiment date range:", sent["date"].min(), sent["date"].max())

merged = trades.merge(sent, on="date", how="left")
print("\nunmatched rows (no sentiment):", merged["sentiment"].isna().sum(), "/", len(merged))

merged["is_close"] = merged["closed_pnl"] != 0
merged["is_win"] = merged["closed_pnl"] > 0
merged["is_loss"] = merged["closed_pnl"] < 0

sent_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
merged["sentiment"] = pd.Categorical(merged["sentiment"], categories=sent_order, ordered=True)

merged.to_parquet("outputs/data/merged_trades.parquet", index=False)
print("\nSaved merged_trades.parquet, shape:", merged.shape)
print(merged["sentiment"].value_counts())
print(merged.head())
