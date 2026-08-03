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
colors = {"Extreme Fear": "#8c1f1f", "Fear": "#e34948", "Neutral": "#a9a8a0",
          "Greed": "#6da7ec", "Extreme Greed": "#184f95"}
FIG = "outputs/figures"


def style_ax(ax):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)


df = pd.read_parquet("outputs/data/merged_trades.parquet")
closes = df[df["is_close"]].copy()

side_perf = closes.groupby(["sentiment", "side"], observed=True)["closed_pnl"].mean().unstack()
side_perf = side_perf.reindex(sent_order)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(sent_order))
w = 0.35
b1 = ax.bar(x - w / 2, side_perf["BUY"], width=w, label="Long (BUY) avg PnL", color="#6da7ec")
b2 = ax.bar(x + w / 2, side_perf["SELL"], width=w, label="Short (SELL) avg PnL", color="#e34948")
ax.set_xticks(x)
ax.set_xticklabels(sent_order, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Avg realized PnL per closing trade (USD)")
ax.set_title("Was the Crowd's Directional Bias Actually Profitable?",
              fontsize=13, fontweight="bold", loc="left", pad=14)
ax.axhline(0, color="#898781", linewidth=0.8)
style_ax(ax)
for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        ax.annotate(f"${h:.0f}", (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
plt.tight_layout()
plt.savefig(f"{FIG}/crowd_bias_correctness.png", dpi=150)
plt.close()

daily = pd.read_csv("outputs/data/daily_agg.csv", parse_dates=["date"]).sort_values("date")
daily_full = daily.set_index("date").asfreq("D")
daily_full["daily_pnl"] = daily_full["daily_pnl"].fillna(0)
daily_full["daily_volume"] = daily_full["daily_volume"].fillna(0)
daily_full["sentiment_value"] = daily_full["sentiment_value"].ffill()

lags = [0, 1, 2, 3, 5, 7, 14]
pnl_corrs = [daily_full["sentiment_value"].shift(k).corr(daily_full["daily_pnl"]) for k in lags]
vol_corrs = [daily_full["sentiment_value"].shift(k).corr(daily_full["daily_volume"]) for k in lags]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(lags, pnl_corrs, marker="o", color="#184f95", label="vs daily PnL", linewidth=2)
ax.plot(lags, vol_corrs, marker="o", color="#e34948", label="vs daily volume", linewidth=2)
ax.axhline(0, color="#898781", linewidth=0.8)
ax.set_xlabel("Sentiment lag (days before the trading day)")
ax.set_ylabel("Correlation coefficient (r)")
ax.set_title("Does Past Sentiment Predict Future Trading Outcomes?",
              fontsize=13, fontweight="bold", loc="left", pad=14)
style_ax(ax)
ax.legend(frameon=False, loc="lower left")
plt.tight_layout()
plt.savefig(f"{FIG}/lagged_sentiment_correlation.png", dpi=150)
plt.close()

beta_df = pd.read_csv("outputs/data/account_sentiment_beta.csv")
beta_df = beta_df.sort_values("sentiment_corr")
bar_colors = ["#8c1f1f" if v < -0.1 else ("#184f95" if v > 0.1 else "#a9a8a0") for v in beta_df["sentiment_corr"]]

fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(range(len(beta_df)), beta_df["sentiment_corr"], color=bar_colors)
ax.set_yticks(range(len(beta_df)))
ax.set_yticklabels([a[:8] + "…" for a in beta_df["account"]], fontsize=7)
ax.axvline(0, color="#898781", linewidth=0.8)
ax.set_xlabel("Correlation: account's daily PnL vs. daily sentiment value")
ax.set_title("Which Accounts Are Pro- vs Counter-Cyclical?",
              fontsize=13, fontweight="bold", loc="left", pad=14)
style_ax(ax)
ax.spines["bottom"].set_visible(True)
plt.tight_layout()
plt.savefig(f"{FIG}/account_sentiment_beta.png", dpi=150)
plt.close()

fi = pd.read_csv("outputs/data/feature_importance.csv", index_col=0)
fi = fi.sort_values("permutation_importance", ascending=True)
label_map = {
    "account_enc": "Which account", "sentiment_value": "Sentiment value",
    "coin_enc": "Which coin", "side_enc": "Long vs short",
    "era_late": "Early vs late era", "log_size_usd": "Trade size",
}
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bar_colors2 = ["#184f95" if idx == "sentiment_value" else "#a9a8a0" for idx in fi.index]
ax.barh([label_map.get(i, i) for i in fi.index], fi["permutation_importance"], color=bar_colors2)
ax.set_xlabel("Permutation importance (drop in ROC-AUC when feature is shuffled)")
ax.set_title("What Actually Predicts Whether a Single Trade Wins?",
              fontsize=13, fontweight="bold", loc="left", pad=14)
style_ax(ax)
ax.spines["bottom"].set_visible(True)
plt.tight_layout()
plt.savefig(f"{FIG}/feature_importance_win.png", dpi=150)
plt.close()

print("saved 4 advanced charts")
