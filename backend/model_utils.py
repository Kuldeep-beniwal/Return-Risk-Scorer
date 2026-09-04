"""
model_utils.py
Loads the deployed model (Random Forest) and the explainer model (Logistic
Regression), and provides a single `score_order()` function that returns a
full scoring result: the deployed model's probability, the flag decision at
the cost-optimal threshold, and exact per-feature reason codes from the
explainer model.

This mirrors scripts/explain_model.py's logic but loads pre-trained
artifacts instead of retraining, so it's fast enough to call per-request.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib

ARTIFACT_DIR = os.environ.get(
    "RISK_MODEL_ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "outputs"),
)

FEATURE_COLS = [
    "category", "price", "discount_pct", "payment_method", "device",
    "past_orders", "past_return_rate", "days_to_deliver",
    "is_gift_wrapped", "size_variants_in_order", "review_score_of_product",
]
CATEGORICAL_COLS = ["category", "payment_method", "device"]


class RiskModel:
    def __init__(self, artifact_dir: str = ARTIFACT_DIR):
        self.rf_pipe = joblib.load(os.path.join(artifact_dir, "best_model.joblib"))
        self.lr_pipe = joblib.load(os.path.join(artifact_dir, "explainer_model.joblib"))

        with open(os.path.join(artifact_dir, "eval_summary.json")) as f:
            eval_summary = json.load(f)
        self.threshold = eval_summary["cost_optimal_threshold"]
        self.cost_false_positive = eval_summary["cost_assumptions"]["cost_false_positive_inr"]
        self.benefit_true_positive = eval_summary["cost_assumptions"]["benefit_true_positive_inr"]

        with open(os.path.join(artifact_dir, "feature_metadata.json")) as f:
            self.feature_metadata = json.load(f)

        # Recover human-readable one-hot feature names + coefficients for
        # exact log-odds decomposition.
        ohe = self.lr_pipe.named_steps["prep"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
        numeric_cols = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
        self.all_feature_names = cat_feature_names + numeric_cols
        self.coefs = self.lr_pipe.named_steps["clf"].coef_[0]
        self.intercept = self.lr_pipe.named_steps["clf"].intercept_[0]

    def validate_order(self, order: dict) -> list:
        """Returns a list of validation error strings (empty list = valid)."""
        errors = []
        for col in FEATURE_COLS:
            if col not in order:
                errors.append(f"Missing field: {col}")
        if errors:
            return errors

        for cat_col, allowed in self.feature_metadata["categorical"].items():
            if order[cat_col] not in allowed:
                errors.append(f"'{order[cat_col]}' is not a valid {cat_col}. Allowed: {allowed}")

        numeric_cols = list(self.feature_metadata["numeric_ranges"].keys())
        for col in numeric_cols:
            try:
                float(order[col])
            except (TypeError, ValueError):
                errors.append(f"Field '{col}' must be numeric.")
        return errors

    def explain(self, order_df: pd.DataFrame, top_n: int = 5) -> dict:
        X_transformed = self.lr_pipe.named_steps["prep"].transform(order_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_transformed = X_transformed[0]

        contributions = X_transformed * self.coefs
        contrib_df = pd.DataFrame({
            "feature": self.all_feature_names,
            "contribution": contributions,
        })
        contrib_df = contrib_df[contrib_df["contribution"].abs() > 1e-9]

        logit = contributions.sum() + self.intercept
        lr_probability = float(1 / (1 + np.exp(-logit)))

        up = (contrib_df[contrib_df["contribution"] > 0]
              .sort_values("contribution", ascending=False)
              .head(top_n))
        down = (contrib_df[contrib_df["contribution"] < 0]
                .sort_values("contribution")
                .head(top_n))

        def _to_records(sub_df):
            return [
                {"feature": _friendly_name(row["feature"]), "log_odds_contribution": round(float(row["contribution"]), 3)}
                for _, row in sub_df.iterrows()
            ]

        return {
            "explainer_probability": round(lr_probability, 4),
            "risk_increasing_factors": _to_records(up),
            "risk_decreasing_factors": _to_records(down),
        }

    def score_order(self, order: dict) -> dict:
        errors = self.validate_order(order)
        if errors:
            return {"errors": errors}

        order_df = pd.DataFrame([{col: order[col] for col in FEATURE_COLS}])
        # Coerce numeric columns to float (form data may arrive as strings)
        for col in self.feature_metadata["numeric_ranges"].keys():
            order_df[col] = order_df[col].astype(float)

        rf_probability = float(self.rf_pipe.predict_proba(order_df)[0, 1])
        flagged = rf_probability >= self.threshold

        explanation = self.explain(order_df)

        return {
            "deployed_model_probability": round(rf_probability, 4),
            "threshold_used": self.threshold,
            "flagged_for_review": bool(flagged),
            "cost_assumptions": {
                "cost_false_positive_inr": self.cost_false_positive,
                "benefit_true_positive_inr": self.benefit_true_positive,
            },
            **explanation,
        }


def _friendly_name(raw_feature_name: str) -> str:
    """Turns one-hot names like 'category_Apparel' into 'category = Apparel'."""
    for cat_col in CATEGORICAL_COLS:
        prefix = f"{cat_col}_"
        if raw_feature_name.startswith(prefix):
            return f"{cat_col} = {raw_feature_name[len(prefix):]}"
    return raw_feature_name
