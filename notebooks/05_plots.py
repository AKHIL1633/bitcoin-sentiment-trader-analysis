import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    "Extreme Fear": "#8c1f1f",
    "Fear": "#e34948",
    "Neutral": "#a9a8a0",
    "Greed": "#6da7ec",
    "Extreme Greed": "#184f95",
}
palette = [colors[s] for s in sent_order]

overall = pd.read_csv("outputs/data/overall_by_sentiment.csv", index_col=0).reindex(sent_order)
daily = pd.read_csv("outputs/data/daily_agg.csv", parse_dates=["date"]).sort_values("date")

FIG = "outputs/figures"


def style_ax(ax):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)


def bar_chart(series, title, ylabel, fname, fmt="{:.0f}", pct=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(sent_order, series.values, color=palette, width=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.set_ylabel(ylabel, fontsize=10)
    style_ax(ax)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    for b, v in zip(bars, series.values):
        label = fmt.format(v * 100) + "%" if pct else fmt.format(v)
        ax.annotate(label, (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color="#0b0b0b")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG}/{fname}", dpi=150)
    plt.close()


bar_chart(overall["win_rate"], "Win Rate by Market Sentiment", "Win rate", "win_rate_by_sentiment.png", pct=True)

bar_chart(overall["total_closed_pnl"] / 1e6, "Total Closed PnL by Market Sentiment",
          "Total closed PnL (USD, millions)", "total_pnl_by_sentiment.png", fmt="${:.2f}M")

bar_chart(overall["pnl_per_dollar_volume_bps"], "Capital Efficiency by Market Sentiment",
          "PnL per $ traded (bps)", "pnl_efficiency_by_sentiment.png", fmt="{:.0f} bps")

bar_chart(overall["profit_factor"], "Profit Factor by Market Sentiment",
          "Profit factor (gross win USD / gross loss USD)", "profit_factor_by_sentiment.png", fmt="{:.1f}x")

bar_chart(overall["total_volume_usd"] / 1e6, "Total Trading Volume by Market Sentiment",
          "Volume (USD, millions)", "volume_by_sentiment.png", fmt="${:.0f}M")

merged = pd.read_parquet("outputs/data/merged_trades.parquet")
opens = merged[merged["direction"].isin(["Open Long", "Open Short"])]
dir_mix = opens.groupby(["sentiment", "direction"], observed=True).size().unstack(fill_value=0)
dir_mix_pct = (dir_mix.div(dir_mix.sum(axis=1), axis=0) * 100).reindex(sent_order)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(sent_order))
w = 0.35
ax.bar(x - w / 2, dir_mix_pct["Open Long"], width=w, label="Open Long", color="#6da7ec")
ax.bar(x + w / 2, dir_mix_pct["Open Short"], width=w, label="Open Short", color="#e34948")
ax.set_xticks(x)
ax.set_xticklabels(sent_order, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("% of position-opening trades")
ax.set_title("Long vs Short Positioning by Market Sentiment", fontsize=13, fontweight="bold", loc="left", pad=14)
ax.axhline(50, color="#898781", linewidth=0.8, linestyle="--")
style_ax(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
plt.tight_layout()
plt.savefig(f"{FIG}/long_short_mix_by_sentiment.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
roll = daily.set_index("date")["daily_pnl"].rolling(14, min_periods=1).mean()
axes[0].plot(daily["date"], roll, color="#2a78d6", linewidth=1.8)
axes[0].axhline(0, color="#898781", linewidth=0.8)
axes[0].set_title("Daily Closed PnL (14-day rolling avg) vs Market Sentiment Over Time",
                   fontsize=13, fontweight="bold", loc="left", pad=14)
axes[0].set_ylabel("Daily PnL (USD)")
style_ax(axes[0])

for s in sent_order:
    sub = daily[daily["sentiment"] == s]
    axes[1].scatter(sub["date"], sub["sentiment_value"], s=6, color=colors[s], label=s)
axes[1].set_ylabel("Fear/Greed value")
axes[1].set_ylim(0, 100)
style_ax(axes[1])
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig(f"{FIG}/daily_pnl_vs_sentiment_timeseries.png", dpi=150)
plt.close()

bar_chart(overall["avg_trade_size_usd"], "Average Trade Size by Market Sentiment",
          "Avg trade size (USD)", "avg_trade_size_by_sentiment.png", fmt="${:.0f}")

print("All figures saved to", FIG)
