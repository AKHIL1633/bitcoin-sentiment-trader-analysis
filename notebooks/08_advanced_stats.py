import pandas as pd
import numpy as np
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
np.random.seed(42)

df = pd.read_parquet("outputs/data/merged_trades.parquet")
sent_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
closes = df[df["is_close"]].copy()
daily = pd.read_csv("outputs/data/daily_agg.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# account activity/size ramped up hard after ~Nov 2024, split era to check the
# sentiment pattern isn't just riding that time trend
CUTOFF = pd.Timestamp("2024-11-01")
df["era"] = np.where(df["date"] < CUTOFF, "early (pre Nov-2024)", "late (Nov-2024 on)")
closes["era"] = np.where(closes["date"] < CUTOFF, "early (pre Nov-2024)", "late (Nov-2024 on)")

print("=== Trade count by era ===")
print(df["era"].value_counts())

for era in ["early (pre Nov-2024)", "late (Nov-2024 on)"]:
    sub = closes[closes["era"] == era]
    g = sub.groupby("sentiment", observed=True)
    tbl = pd.DataFrame({
        "n_closes": g.size(),
        "win_rate": g["is_win"].mean(),
        "avg_pnl": g["closed_pnl"].mean(),
    }).reindex(sent_order)
    vol = df[df["era"] == era].groupby("sentiment", observed=True)["size_usd"].sum().reindex(sent_order)
    pnl_sum = g["closed_pnl"].sum().reindex(sent_order)
    tbl["pnl_per_dollar_bps"] = pnl_sum / vol * 1e4
    print(f"\n=== Regime metrics — {era} ===")
    print(tbl)

daily_full = daily.set_index("date").asfreq("D")
daily_full["daily_pnl"] = daily_full["daily_pnl"].fillna(0)
daily_full["daily_volume"] = daily_full["daily_volume"].fillna(0)
daily_full["daily_trades"] = daily_full["daily_trades"].fillna(0)
daily_full["sentiment_value"] = daily_full["sentiment_value"].ffill()

print("\n=== Lagged correlation: sentiment_value(t-k) vs daily_pnl(t) ===")
for k in [0, 1, 2, 3, 5, 7, 14]:
    lagged = daily_full["sentiment_value"].shift(k)
    c_pnl = lagged.corr(daily_full["daily_pnl"])
    c_vol = lagged.corr(daily_full["daily_volume"])
    c_trd = lagged.corr(daily_full["daily_trades"])
    print(f"lag={k:>2}d   pnl r={c_pnl:+.3f}   volume r={c_vol:+.3f}   trades r={c_trd:+.3f}")

groups = [closes.loc[closes["sentiment"] == s, "closed_pnl"].values for s in sent_order]
h_stat, p_kw = stats.kruskal(*groups)
print(f"\n=== Kruskal-Wallis across all 5 sentiment regimes (closed PnL) ===")
print(f"H={h_stat:.2f}, p={p_kw:.3g}")

n_total = sum(len(g) for g in groups)
eps_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
print(f"epsilon-squared effect size: {eps_sq:.4f} (0.01 small / 0.08 medium / 0.26 large — Cohen's convention)")


def bootstrap_ci(values, stat_fn, n_boot=5000, ci=95):
    values = np.asarray(values)
    boots = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[np.random.randint(0, n, n)]
        boots[i] = stat_fn(sample)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


print("\n=== Bootstrap 95% CIs (5000 resamples) ===")
for s in sent_order:
    vals = closes.loc[closes["sentiment"] == s, "closed_pnl"].values
    wins = (vals > 0).astype(float)
    wr_lo, wr_hi = bootstrap_ci(wins, np.mean)
    pnl_lo, pnl_hi = bootstrap_ci(vals, np.mean)
    print(f"{s:>14}: win_rate 95% CI [{wr_lo:.3f}, {wr_hi:.3f}]   avg_pnl 95% CI [${pnl_lo:.2f}, ${pnl_hi:.2f}]   n={len(vals)}")

print("\n=== Avg closed PnL by side, within each sentiment regime ===")
side_perf = closes.groupby(["sentiment", "side"], observed=True).agg(
    n=("closed_pnl", "size"), avg_pnl=("closed_pnl", "mean"), win_rate=("is_win", "mean")
).reindex(sent_order, level=0)
print(side_perf)

acct_daily = df.groupby(["account", "date"], observed=True).agg(
    pnl=("closed_pnl", "sum"), sentiment_value=("sentiment_value", "first")
).reset_index()

betas = []
for acct, g in acct_daily.groupby("account"):
    if g["date"].nunique() < 10:
        continue
    r = g["sentiment_value"].corr(g["pnl"])
    betas.append((acct, r, g["pnl"].sum(), g["date"].nunique()))
beta_df = pd.DataFrame(betas, columns=["account", "sentiment_corr", "total_pnl", "n_days"]).sort_values("sentiment_corr", ascending=False)
print("\n=== Per-account correlation of daily PnL with sentiment value ===")
print(beta_df.to_string(index=False))
beta_df.to_csv("outputs/data/account_sentiment_beta.csv", index=False)

print(f"\nAccounts positively correlated with sentiment (pro-cyclical, do better in greed): {(beta_df['sentiment_corr'] > 0.1).sum()}")
print(f"Accounts negatively correlated with sentiment (counter-cyclical, do better in fear): {(beta_df['sentiment_corr'] < -0.1).sum()}")
print(f"Accounts roughly sentiment-neutral: {(beta_df['sentiment_corr'].abs() <= 0.1).sum()}")
