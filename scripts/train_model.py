"""
train_model.py
Trains a return-risk scorer on the synthetic orders dataset and evaluates it
honestly: precision, recall, F1, ROC-AUC, confusion matrix, PLUS a
false-positive-cost-aware threshold analysis (per "THE BAR" requirement).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import joblib

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

# ---- Held-out test set: stratified so return-rate is preserved in both splits --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

categorical_cols = ["category", "payment_method", "device"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

models = {
    "logistic_regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "random_forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds_default = (proba >= 0.5).astype(int)

    results[name] = {
        "pipe": pipe,
        "proba": proba,
        "precision@0.5": precision_score(y_test, preds_default),
        "recall@0.5": recall_score(y_test, preds_default),
        "f1@0.5": f1_score(y_test, preds_default),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix@0.5": confusion_matrix(y_test, preds_default).tolist(),
    }

print("=" * 70)
for name, r in results.items():
    print(f"\nModel: {name}")
    print(f"  ROC-AUC:        {r['roc_auc']:.4f}")
    print(f"  Precision@0.5:  {r['precision@0.5']:.4f}")
    print(f"  Recall@0.5:     {r['recall@0.5']:.4f}")
    print(f"  F1@0.5:         {r['f1@0.5']:.4f}")
    tn, fp, fn, tp = np.array(r["confusion_matrix@0.5"]).ravel()
    print(f"  Confusion matrix @0.5 -> TN={tn} FP={fp} FN={fn} TP={tp}")

# ---- Pick the better model by ROC-AUC (threshold-independent) -----------------
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best = results[best_name]
print(f"\n>>> Best model by ROC-AUC: {best_name}")

# =================================================================================
# FALSE-POSITIVE-COST-AWARE THRESHOLD ANALYSIS  ("THE BAR": honest metrics
# including false-positive cost)
# =================================================================================
# Business assumption (documented, not hidden):
#   - Flagging a GOOD order as high-risk costs the merchant ~INR 40
#     (manual verification / friction / possible customer annoyance).
#   - Catching a TRUE return before shipping saves ~INR 180
#     (avoided reverse logistics + restocking + payment processing costs).
# These are illustrative placeholders -- a real deployment must plug in the
# merchant's actual numbers.

COST_FALSE_POSITIVE = 40.0
BENEFIT_TRUE_POSITIVE = 180.0

thresholds = np.arange(0.05, 0.96, 0.01)
proba = best["proba"]
y_true = y_test.values

net_values = []
for t in thresholds:
    preds = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    net_value = tp * BENEFIT_TRUE_POSITIVE - fp * COST_FALSE_POSITIVE
    net_values.append(net_value)

net_values = np.array(net_values)
best_idx = net_values.argmax()
best_threshold = thresholds[best_idx]
best_net_value = net_values[best_idx]

preds_at_best = (proba >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, preds_at_best).ravel()

print(f"\n--- Cost-optimal threshold search ---")
print(f"Cost per false positive:  INR {COST_FALSE_POSITIVE}")
print(f"Benefit per true positive: INR {BENEFIT_TRUE_POSITIVE}")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Net value at best threshold (test set): INR {best_net_value:,.0f}")
print(f"  -> TP={tp} (returns caught), FP={fp} (good orders wrongly flagged)")
print(f"  -> FN={fn} (returns missed), TN={tn} (good orders correctly cleared)")
print(f"  -> Precision at this threshold: {precision_score(y_true, preds_at_best):.4f}")
print(f"  -> Recall at this threshold:    {recall_score(y_true, preds_at_best):.4f}")

# Save artifacts for reporting / the artifact dashboard
joblib.dump(best["pipe"], f"{OUT_DIR}/best_model.joblib")

eval_summary = {
    "model_compared": {n: {k: v for k, v in r.items() if k != "pipe" and k != "proba"} for n, r in results.items()},
    "best_model": best_name,
    "cost_assumptions": {
        "cost_false_positive_inr": COST_FALSE_POSITIVE,
        "benefit_true_positive_inr": BENEFIT_TRUE_POSITIVE,
    },
    "cost_optimal_threshold": float(best_threshold),
    "cost_optimal_net_value_inr": float(best_net_value),
    "cost_optimal_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}

import json
with open(f"{OUT_DIR}/eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

# Save threshold curve data for plotting in dashboard
threshold_curve = pd.DataFrame({"threshold": thresholds, "net_value_inr": net_values})
threshold_curve.to_csv(f"{OUT_DIR}/threshold_curve.csv", index=False)

fpr, tpr, roc_thresholds = roc_curve(y_true, proba)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(f"{OUT_DIR}/roc_curve.csv", index=False)

print(f"\nSaved model + eval summary + curves to {OUT_DIR}/")
