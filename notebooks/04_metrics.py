import pandas as pd
import numpy as np
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_parquet("outputs/data/merged_trades.parquet")
sent_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

overall = df.groupby("sentiment", observed=True).agg(
    n_trades=("closed_pnl", "size"),
    n_closes=("is_close", "sum"),
    total_volume_usd=("size_usd", "sum"),
    avg_trade_size_usd=("size_usd", "mean"),
    total_closed_pnl=("closed_pnl", "sum"),
    total_fees=("fee", "sum"),
).reindex(sent_order)

closes = df[df["is_close"]]
win_stats = closes.groupby("sentiment", observed=True).agg(
    win_rate=("is_win", "mean"),
    avg_win=("closed_pnl", lambda s: s[s > 0].mean()),
    avg_loss=("closed_pnl", lambda s: s[s < 0].mean()),
    avg_pnl_per_close=("closed_pnl", "mean"),
    median_pnl_per_close=("closed_pnl", "median"),
).reindex(sent_order)

overall = overall.join(win_stats)
overall["net_pnl_after_fees"] = overall["total_closed_pnl"] - overall["total_fees"]
overall["pnl_per_dollar_volume_bps"] = overall["total_closed_pnl"] / overall["total_volume_usd"] * 1e4
overall["profit_factor"] = -win_stats["avg_win"] * closes.groupby("sentiment", observed=True)["is_win"].sum() / \
    (win_stats["avg_loss"] * closes.groupby("sentiment", observed=True).apply(lambda g: (g["closed_pnl"] < 0).sum(), include_groups=False))

print("=== OVERALL METRICS BY SENTIMENT ===")
print(overall.T)
overall.to_csv("outputs/data/overall_by_sentiment.csv")

side_mix = df.groupby(["sentiment", "side"], observed=True).size().unstack(fill_value=0)
side_mix_pct = side_mix.div(side_mix.sum(axis=1), axis=0) * 100
print("\n=== BUY/SELL MIX % BY SENTIMENT ===")
print(side_mix_pct.reindex(sent_order))

# Direction-based open long/short
opens = df[df["direction"].isin(["Open Long", "Open Short"])]
dir_mix = opens.groupby(["sentiment", "direction"], observed=True).size().unstack(fill_value=0)
dir_mix_pct = dir_mix.div(dir_mix.sum(axis=1), axis=0) * 100
print("\n=== OPEN LONG/SHORT MIX % BY SENTIMENT (of position-opening trades) ===")
print(dir_mix_pct.reindex(sent_order))

daily = df.groupby("date").agg(
    daily_pnl=("closed_pnl", "sum"),
    daily_volume=("size_usd", "sum"),
    daily_trades=("closed_pnl", "size"),
    sentiment_value=("sentiment_value", "first"),
    sentiment=("sentiment", "first"),
).reset_index().sort_values("date")

corr_pnl = daily["sentiment_value"].corr(daily["daily_pnl"])
corr_vol = daily["sentiment_value"].corr(daily["daily_volume"])
corr_trades = daily["sentiment_value"].corr(daily["daily_trades"])
print(f"\nCorrelation sentiment_value vs daily_pnl: {corr_pnl:.4f}")
print(f"Correlation sentiment_value vs daily_volume: {corr_vol:.4f}")
print(f"Correlation sentiment_value vs daily_trades: {corr_trades:.4f}")

daily.to_csv("outputs/data/daily_agg.csv", index=False)

fear_pnl = closes[closes["sentiment"].isin(["Fear", "Extreme Fear"])]["closed_pnl"]
greed_pnl = closes[closes["sentiment"].isin(["Greed", "Extreme Greed"])]["closed_pnl"]
t_stat, p_val = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
print(f"\nMann-Whitney U test Fear vs Greed closed PnL: U={t_stat:.1f}, p={p_val:.4g}")
print(f"Fear mean/median PnL: {fear_pnl.mean():.2f} / {fear_pnl.median():.2f}  (n={len(fear_pnl)})")
print(f"Greed mean/median PnL: {greed_pnl.mean():.2f} / {greed_pnl.median():.2f}  (n={len(greed_pnl)})")

acct = df.groupby(["account", "sentiment"], observed=True)["closed_pnl"].sum().unstack(fill_value=0).reindex(columns=sent_order)
acct["total_pnl"] = acct.sum(axis=1)
acct = acct.sort_values("total_pnl", ascending=False)
print("\n=== TOP 10 ACCOUNTS: PnL by sentiment ===")
print(acct.head(10))
acct.to_csv("outputs/data/account_pnl_by_sentiment.csv")

top_coins = df.groupby(["sentiment", "coin"], observed=True)["size_usd"].sum().reset_index()
top5 = top_coins.sort_values(["sentiment", "size_usd"], ascending=[True, False]).groupby("sentiment", observed=True).head(5)
print("\n=== TOP 5 COINS BY VOLUME PER SENTIMENT ===")
print(top5.to_string(index=False))
