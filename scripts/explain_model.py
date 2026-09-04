"""
explain_model.py
Explainability layer for the return-risk scorer.

Two complementary views:

1. GLOBAL IMPORTANCE (which features matter most, across all orders)
   - Permutation importance on the deployed model (Random Forest), computed on
     the held-out test set. Model-agnostic and honest: it measures how much
     ROC-AUC actually drops when a feature is shuffled, not an internal proxy.

2. PER-ORDER REASON CODES (why THIS order got flagged)
   - A parallel Logistic Regression model is used purely as an explanation
     layer: its coefficients decompose a prediction into an exact sum of
     per-feature log-odds contributions (contributions literally add up to
     the predicted logit -- no approximation).
   - This mirrors how regulated credit/risk systems produce "reason codes":
     the production score can come from a stronger black-box model, while a
     transparent linear model explains *why*.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib
import json

DATA_PATH = "/home/claude/return_risk_scorer/data/orders.csv"
OUT_DIR = "/home/claude/return_risk_scorer/outputs"

df = pd.read_csv(DATA_PATH)

feature_cols = [
    "category", "price", "discount_pct", "payment_method", "device",
    "past_orders", "past_return_rate", "days_to_deliver",
    "is_gift_wrapped", "size_variants_in_order", "review_score_of_product",
]
target_col = "returned"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

categorical_cols = ["category", "payment_method", "device"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# ---- Load the deployed model (Random Forest, chosen in train_model.py) --------
rf_pipe = joblib.load(f"{OUT_DIR}/best_model.joblib")

# ---- 1. GLOBAL IMPORTANCE: permutation importance on RAW features -------------
# Passing the full pipeline means each ORIGINAL column (e.g. "category",
# "past_return_rate") is shuffled -- so importance is reported in terms
# merchants actually understand, not one-hot dummy columns.
perm_result = permutation_importance(
    rf_pipe, X_test, y_test, scoring="roc_auc",
    n_repeats=15, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False)

print("=" * 70)
print("GLOBAL FEATURE IMPORTANCE (permutation importance, drop in ROC-AUC)")
print("=" * 70)
print(importance_df.to_string(index=False))
importance_df.to_csv(f"{OUT_DIR}/global_feature_importance.csv", index=False)

# ---- 2. Fit the explanation-layer Logistic Regression -------------------------
lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
lr_pipe.fit(X_train, y_train)

# Recover human-readable feature names after one-hot encoding
ohe = lr_pipe.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = list(cat_feature_names) + numeric_cols

coefs = lr_pipe.named_steps["clf"].coef_[0]
intercept = lr_pipe.named_steps["clf"].intercept_[0]

coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": coefs})
coef_df = coef_df.sort_values("coefficient", ascending=False)
coef_df.to_csv(f"{OUT_DIR}/reason_code_coefficients.csv", index=False)

# Persist the explainer pipeline + feature metadata so a backend service can
# load everything at startup without retraining on every request.
joblib.dump(lr_pipe, f"{OUT_DIR}/explainer_model.joblib")

feature_metadata = {
    "categorical": {
        "category": sorted(df["category"].unique().tolist()),
        "payment_method": sorted(df["payment_method"].unique().tolist()),
        "device": sorted(df["device"].unique().tolist()),
    },
    "numeric_ranges": {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in numeric_cols
    },
}
with open(f"{OUT_DIR}/feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

print("\n" + "=" * 70)
print("TOP RISK-INCREASING FACTORS (logistic regression, log-odds coefficient)")
print("=" * 70)
print(coef_df.head(8).to_string(index=False))
print("\nTOP RISK-DECREASING FACTORS")
print(coef_df.tail(8).to_string(index=False))


def explain_order(order_row: pd.Series, top_n: int = 3) -> dict:
    """
    Returns an EXACT decomposition of this order's predicted log-odds into
    per-feature contributions (they sum to the raw logit). Also returns the
    top_n risk-increasing and top_n risk-decreasing factors in plain language.
    """
    order_df = pd.DataFrame([order_row[feature_cols]])
    X_transformed = lr_pipe.named_steps["prep"].transform(order_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = X_transformed[0]

    contributions = X_transformed * coefs
    contrib_df = pd.DataFrame({
        "feature": all_feature_names,
        "contribution": contributions,
    })
    # Drop near-zero one-hot columns that weren't active for this order (no signal)
    contrib_df = contrib_df[contrib_df["contribution"].abs() > 1e-9]
    contrib_df = contrib_df.sort_values("contribution", ascending=False)

    logit = contributions.sum() + intercept
    probability = 1 / (1 + np.exp(-logit))

    # Only genuinely risk-increasing (positive) / risk-decreasing (negative)
    # contributions qualify -- an order with few active risk factors should
    # show fewer than top_n rather than padding with near-zero noise.
    up = contrib_df[contrib_df["contribution"] > 0].head(top_n).to_dict("records")
    down = contrib_df[contrib_df["contribution"] < 0].sort_values("contribution").head(top_n).to_dict("records")

    return {
        "order_id": order_row["order_id"],
        "predicted_return_probability_explainer_lr": round(float(probability), 4),
        "top_risk_increasing_factors": up,
        "top_risk_decreasing_factors": down,
    }


def human_readable_explanation(order_row: pd.Series, rf_probability: float = None) -> str:
    result = explain_order(order_row)
    header = f"Order {result['order_id']}:"
    if rf_probability is not None:
        header += f" deployed model (RF) score = {rf_probability:.1%}"
    header += f"  |  explainer model (LR) score = {result['predicted_return_probability_explainer_lr']:.1%}"
    lines = [header, "  Risk-increasing factors:"]
    if result["top_risk_increasing_factors"]:
        for f in result["top_risk_increasing_factors"]:
            lines.append(f"    + {f['feature']}  (contribution: +{f['contribution']:.2f} log-odds)")
    else:
        lines.append("    (none material)")
    lines.append("  Risk-decreasing factors:")
    if result["top_risk_decreasing_factors"]:
        for f in result["top_risk_decreasing_factors"]:
            lines.append(f"    - {f['feature']}  (contribution: {f['contribution']:.2f} log-odds)")
    else:
        lines.append("    (none material)")
    return "\n".join(lines)


# ---- Demonstrate on a few real orders from the test set -----------------------
print("\n" + "=" * 70)
print("PER-ORDER REASON CODES (sample orders)")
print("=" * 70)

sample_df = df.loc[X_test.index].copy()
sample_df["true_label"] = y_test.values
sample_df["predicted_proba_rf"] = rf_pipe.predict_proba(X_test)[:, 1]

# One clearly high-risk order and one clearly low-risk order, for contrast
highest = sample_df.sort_values("predicted_proba_rf", ascending=False).iloc[0]
lowest = sample_df.sort_values("predicted_proba_rf", ascending=True).iloc[0]

for label, row in [("HIGH-RISK EXAMPLE", highest), ("LOW-RISK EXAMPLE", lowest)]:
    print(f"\n--- {label} ---")
    print(human_readable_explanation(row, rf_probability=row["predicted_proba_rf"]))

# Save a batch of reason codes for the whole test set (for a demo/dashboard)
all_explanations = []
for _, row in sample_df.head(200).iterrows():  # cap for speed/output size
    all_explanations.append(explain_order(row))

with open(f"{OUT_DIR}/sample_reason_codes.json", "w") as f:
    json.dump(all_explanations, f, indent=2)

print(f"\nSaved global importance, coefficients, and sample reason codes to {OUT_DIR}/")
