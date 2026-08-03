import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
np.random.seed(42)

df = pd.read_parquet("outputs/data/merged_trades.parquet")
closes = df[df["is_close"]].copy()

# ----- feature engineering -----
closes["log_size_usd"] = np.log1p(closes["size_usd"])
closes["era_late"] = (closes["date"] >= pd.Timestamp("2024-11-01")).astype(int)

top_coins = closes["coin"].value_counts().head(15).index
closes["coin_bucket"] = np.where(closes["coin"].isin(top_coins), closes["coin"], "OTHER")

le_coin = LabelEncoder()
le_acct = LabelEncoder()
le_side = LabelEncoder()

X = pd.DataFrame({
    "sentiment_value": closes["sentiment_value"].astype(float),
    "log_size_usd": closes["log_size_usd"],
    "side_enc": le_side.fit_transform(closes["side"]),
    "coin_enc": le_coin.fit_transform(closes["coin_bucket"]),
    "account_enc": le_acct.fit_transform(closes["account"]),
    "era_late": closes["era_late"],
})
y = closes["is_win"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ----- Random Forest classifier -----
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=20,
                             class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
pred_proba = rf.predict_proba(X_test)[:, 1]
pred = rf.predict(X_test)
print("=== Random Forest: predicting is_win from trade features ===")
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, pred_proba):.3f}")
print(f"Baseline (always predict majority class) accuracy: {max(y_test.mean(), 1-y_test.mean()):.3f}")

print("\n--- Impurity-based feature importance (biased toward high-cardinality features) ---")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp)

print("\n--- Permutation importance on held-out test set (fairer ranking) ---")
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="roc_auc")
perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print(perm_imp)

# ----- Logistic regression for interpretable, signed coefficients -----
X_scaled = X.copy()
scaler = StandardScaler()
X_scaled[["sentiment_value", "log_size_usd"]] = scaler.fit_transform(X[["sentiment_value", "log_size_usd"]])
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
logit = LogisticRegression(max_iter=2000, class_weight="balanced")
logit.fit(Xtr, ytr)
print("\n=== Logistic regression coefficients (standardized sentiment_value & log_size_usd) ===")
coef = pd.Series(logit.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(coef)
print(f"Logistic ROC-AUC on test: {roc_auc_score(yte, logit.predict_proba(Xte)[:,1]):.3f}")

# ----- sanity: model with sentiment ONLY -----
X_sent_only = X[["sentiment_value"]]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_sent_only, y, test_size=0.25, random_state=42, stratify=y)
logit2 = LogisticRegression(max_iter=1000, class_weight="balanced")
logit2.fit(Xtr2, ytr2)
print(f"\nROC-AUC using sentiment_value ALONE (no other features): {roc_auc_score(yte2, logit2.predict_proba(Xte2)[:,1]):.3f}")
print("(0.50 = no predictive power beyond chance; compare to full-feature AUC above)")

results = pd.DataFrame({
    "impurity_importance": imp,
    "permutation_importance": perm_imp,
    "logit_coef_abs": coef.abs(),
}).sort_values("permutation_importance", ascending=False)
results.to_csv("outputs/data/feature_importance.csv")
