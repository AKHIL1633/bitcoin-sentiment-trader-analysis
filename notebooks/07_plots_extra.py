import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial"],
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})

sent_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
colors = {
    "Extreme Fear": "#8c1f1f", "Fear": "#e34948", "Neutral": "#a9a8a0",
    "Greed": "#6da7ec", "Extreme Greed": "#184f95",
}
palette = [colors[s] for s in sent_order]
FIG = "outputs/figures"


def style_ax(ax):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)


df = pd.read_parquet("outputs/data/merged_trades.parquet")
closes = df[df["is_close"]]

# ---- Trades per active day by sentiment ----
n_days = df.drop_duplicates("date").groupby("sentiment", observed=True).size().reindex(sent_order)
trades_per_day = (df.groupby("sentiment", observed=True).size().reindex(sent_order) / n_days)

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(sent_order, trades_per_day.values, color=palette, width=0.6)
ax.set_title("Trading Activity by Market Sentiment", fontsize=13, fontweight="bold", loc="left", pad=14)
ax.set_ylabel("Avg trades per active day (all accounts)")
style_ax(ax)
for b, v in zip(bars, trades_per_day.values):
    ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9)
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/trades_per_day_by_sentiment.png", dpi=150)
plt.close()

# ---- Top/bottom coins by total closed PnL ----
coin_pnl = closes.groupby("coin").agg(n=("closed_pnl", "size"), total_pnl=("closed_pnl", "sum"))
coin_pnl = coin_pnl[coin_pnl["n"] >= 100]
top10 = coin_pnl.sort_values("total_pnl", ascending=False).head(8)
bot10 = coin_pnl.sort_values("total_pnl", ascending=True).head(8)
combo = pd.concat([bot10.sort_values("total_pnl"), top10.sort_values("total_pnl")])

fig, ax = plt.subplots(figsize=(8, 7))
bar_colors = ["#8c1f1f" if v < 0 else "#184f95" for v in combo["total_pnl"]]
ax.barh(combo.index.astype(str), combo["total_pnl"] / 1e3, color=bar_colors)
ax.axvline(0, color="#898781", linewidth=0.8)
ax.set_xlabel("Total closed PnL (USD, thousands)")
ax.set_title("Best & Worst Performing Coins (min. 100 closed trades)",
              fontsize=13, fontweight="bold", loc="left", pad=14)
style_ax(ax)
ax.spines["bottom"].set_visible(True)
plt.tight_layout()
plt.savefig(f"{FIG}/coin_pnl_leaders_laggards.png", dpi=150)
plt.close()

print("done")
