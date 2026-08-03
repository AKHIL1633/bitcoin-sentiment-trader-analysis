import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

df = pd.read_parquet("outputs/data/merged_trades.parquet")
sent_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
closes = df[df["is_close"]].copy()

# aligned = long during Fear/Extreme Fear/Neutral, short during Greed/Extreme Greed,
# matching the cohort's own revealed positioning bias
aligned_side = {
    "Extreme Fear": "BUY", "Fear": "BUY", "Neutral": "BUY",
    "Greed": "SELL", "Extreme Greed": "SELL",
}
closes["aligned"] = closes.apply(lambda r: r["side"] == aligned_side[r["sentiment"]], axis=1)

print("=== Actual trades: aligned (contrarian-consistent) vs misaligned side, by regime ===")
tbl = closes.groupby(["sentiment", "aligned"], observed=True).agg(
    n=("closed_pnl", "size"), total_pnl=("closed_pnl", "sum"), avg_pnl=("closed_pnl", "mean")
).reindex(sent_order, level=0)
print(tbl)

total_actual = closes["closed_pnl"].sum()
total_aligned_pnl = closes.loc[closes["aligned"], "closed_pnl"].sum()
total_misaligned_pnl = closes.loc[~closes["aligned"], "closed_pnl"].sum()
n_aligned = closes["aligned"].sum()
n_total = len(closes)

print(f"\nTotal realized PnL (all closes): ${total_actual:,.0f}")
print(f"  From aligned-side trades:    ${total_aligned_pnl:,.0f}  ({n_aligned:,} trades, {n_aligned/n_total*100:.1f}% of closes)")
print(f"  From misaligned-side trades: ${total_misaligned_pnl:,.0f}  ({n_total-n_aligned:,} trades, {(n_total-n_aligned)/n_total*100:.1f}% of closes)")
print(f"  Aligned trades' share of PnL: {total_aligned_pnl/total_actual*100:.1f}% from {n_aligned/n_total*100:.1f}% of trades")

# what if only aligned-side trades had been taken, same size, actual realized PnL
print("\n=== Backtest A: what-if only aligned-side trades had been taken ===")
for s in sent_order:
    sub = closes[closes["sentiment"] == s]
    actual = sub["closed_pnl"].sum()
    aligned_only = sub.loc[sub["aligned"], "closed_pnl"].sum()
    aligned_share_of_volume = sub["aligned"].mean()
    print(f"{s:>14}: actual total ${actual:>12,.0f}  |  aligned-only total ${aligned_only:>12,.0f}  "
          f"|  aligned = {aligned_share_of_volume*100:4.1f}% of that regime's closes")

# scale position size by regime (2x Extreme Greed ... 0.5x Extreme Fear) vs flat 1x baseline
# simplification: assumes the same trades occur at scaled size, no slippage/liquidity impact modeled
scale = {"Extreme Fear": 0.5, "Fear": 0.7, "Neutral": 1.0, "Greed": 1.3, "Extreme Greed": 2.0}
closes["scaled_pnl"] = closes.apply(lambda r: r["closed_pnl"] * scale[r["sentiment"]], axis=1)

baseline_total = closes["closed_pnl"].sum()
scaled_total = closes["scaled_pnl"].sum()
weighted_avg_scale = sum(scale[s] * (closes["sentiment"] == s).sum() for s in sent_order) / len(closes)
scaled_total_normalized = scaled_total / weighted_avg_scale

print(f"\n=== Backtest B: sentiment-scaled position sizing (simplified, no liquidity impact modeled) ===")
print(f"Baseline (flat 1x sizing) total realized PnL: ${baseline_total:,.0f}")
print(f"Sentiment-scaled sizing total realized PnL (normalized to same avg capital): ${scaled_total_normalized:,.0f}")
print(f"Uplift: {(scaled_total_normalized/baseline_total - 1)*100:+.1f}%")

# what if Fear (least efficient regime) had just been skipped entirely
fear_mask = closes["sentiment"].isin(["Fear"])
print(f"\n=== Backtest C: what if Fear-regime trades were skipped entirely ===")
print(f"PnL given up by skipping Fear: ${closes.loc[fear_mask, 'closed_pnl'].sum():,.0f} "
      f"({closes.loc[fear_mask,'closed_pnl'].sum()/baseline_total*100:.1f}% of total PnL)")
fear_vol = df.loc[df['sentiment']=='Fear', 'size_usd'].sum()
total_vol = df['size_usd'].sum()
print(f"Capital/volume avoided by skipping Fear: ${fear_vol:,.0f} ({fear_vol/total_vol*100:.1f}% of total volume)")
print("-> Fear's PnL share is roughly proportional to its volume share, so skipping it outright "
      "doesn't obviously help; the gain in A/B comes from sizing/side discipline within Fear, not avoiding it.")
