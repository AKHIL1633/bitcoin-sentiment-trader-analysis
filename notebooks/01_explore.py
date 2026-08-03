import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

sent = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data.csv")

print("=== SENTIMENT ===")
print(sent.shape)
print(sent.dtypes)
print(sent.head())
print(sent["classification"].value_counts())
print("date range:", sent["date"].min(), sent["date"].max())

print("\n=== TRADES ===")
print(trades.shape)
print(trades.dtypes)
print(trades.head())
print("\nnulls:\n", trades.isnull().sum())
print("\nunique accounts:", trades["Account"].nunique())
print("unique coins:", trades["Coin"].nunique())
print("side values:", trades["Side"].unique())
print("direction values:", trades["Direction"].unique())
print("crossed values:", trades["Crossed"].unique())

# timestamp parse check
print("\nTimestamp IST sample:", trades["Timestamp IST"].iloc[0])
print("Timestamp sample:", trades["Timestamp"].iloc[0])
