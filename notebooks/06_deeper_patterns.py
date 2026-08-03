import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_parquet("outputs/data/merged_trades.parquet")
sent_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
closes = df[df["is_close"]]

acct_pnl = df.groupby(["account", "sentiment"], observed=True)["closed_pnl"].sum().unstack(fill_value=0).reindex(columns=sent_order)
n_regimes_present = (acct_pnl != 0).sum(axis=1)
profitable_all = (acct_pnl > 0).sum(axis=1)
print("Accounts profitable in all 5 sentiment buckets:", (profitable_all == 5).sum(), "/", len(acct_pnl))
print("Accounts profitable in 0 buckets (net loser everywhere):", (profitable_all == 0).sum())
print("\nDistribution of #regimes profitable:")
print(profitable_all.value_counts().sort_index())

eg = closes[closes["sentiment"] == "Extreme Greed"]
eg_by_acct = eg.groupby("account")["closed_pnl"].sum().sort_values(ascending=False)
print("\nExtreme Greed total closed PnL:", eg_by_acct.sum())
print("Top 5 accounts' share of Extreme Greed PnL:")
top5_share = eg_by_acct.head(5).sum() / eg_by_acct.sum()
print(eg_by_acct.head(5))
print(f"Top 5 accounts = {top5_share*100:.1f}% of Extreme Greed PnL (n_accounts active={eg_by_acct.shape[0]})")

fear_g = closes[closes["sentiment"].isin(["Fear"])]
fear_by_acct = fear_g.groupby("account")["closed_pnl"].sum().sort_values(ascending=False)
print(f"\nFear total closed PnL: {fear_by_acct.sum():.0f}, top5 share: {fear_by_acct.head(5).sum()/fear_by_acct.sum()*100:.1f}%")

daily = pd.read_csv("outputs/data/daily_agg.csv", parse_dates=["date"])
n_days_per_sent = df.drop_duplicates("date").groupby("sentiment", observed=True).size().reindex(sent_order)
trades_per_day = df.groupby("sentiment", observed=True).size().reindex(sent_order) / n_days_per_sent
print("\nAvg trades per active day, by sentiment:")
print(trades_per_day)

liq = df[df["direction"].str.contains("Liquidat", case=False, na=False)]
liq_counts = liq.groupby("sentiment", observed=True).size().reindex(sent_order).fillna(0)
total_counts = df.groupby("sentiment", observed=True).size().reindex(sent_order)
print("\nLiquidation events by sentiment:")
print(liq_counts)
print("Liquidation rate (per 10k trades):")
print((liq_counts / total_counts * 10000).round(2))

size_stats = df.groupby("sentiment", observed=True)["size_usd"].agg(["mean", "median", "std", lambda s: s.quantile(0.95)]).reindex(sent_order)
size_stats.columns = ["mean", "median", "std", "p95"]
print("\nTrade size (USD) distribution by sentiment:")
print(size_stats)

coin_share = df.groupby(["sentiment", "coin"], observed=True)["size_usd"].sum()
btc_share = coin_share.xs("BTC", level="coin") / df.groupby("sentiment", observed=True)["size_usd"].sum()
print("\nBTC share of volume by sentiment:")
print(btc_share.reindex(sent_order))

# top gainers/losers coins by avg pnl per trade, filtered to reasonably traded coins
coin_pnl = closes.groupby("coin").agg(n=("closed_pnl", "size"), total_pnl=("closed_pnl", "sum"), avg_pnl=("closed_pnl", "mean"))
coin_pnl = coin_pnl[coin_pnl["n"] >= 100].sort_values("total_pnl", ascending=False)
print("\nTop 10 coins by total closed PnL (n>=100 closes):")
print(coin_pnl.head(10))
print("\nBottom 10 coins by total closed PnL (n>=100 closes):")
print(coin_pnl.tail(10))
