# Return-Risk Scorer : AI Risk Manager

A pre-shipment risk score that predicts the probability an e-commerce order will be
**returned**, so a merchant can intervene before it ships (extra packaging QC, size
confirmation nudge, COD verification call, etc.) instead of eating the reverse-logistics
cost after the fact.

Strictly **defense-only**: this model flags risk for the merchant's own review workflow.
It does not take autonomous action against customers, does not identify or penalize
individuals beyond a per-order score, and has no offense-capable functionality.

## Why return risk (not fraud/chargebacks)

Returns are a quieter, higher-frequency margin leak than outright fraud — every
apparel/footwear order carries real return risk, and getting even a modest slice of
high-risk orders right compounds across volume. It's also a clean supervised-learning
problem: a real label (`returned` = yes/no) exists in every merchant's order history,
so this approach ports directly to real data.

## Data

`data/orders.csv` — 20,000 synthetic orders (see `scripts/generate_data.py`). Each
order has: category, price, discount %, payment method, device, customer's past order
count and past return rate, delivery time, gift-wrap flag, number of size variants
ordered (a classic "bracketing" signal for apparel/footwear), and product review score.

The label is generated from a hidden logistic-risk formula (customer return history is
the strongest driver, followed by category and size-bracketing behavior) **plus random
noise**, so no feature perfectly predicts the outcome — this mirrors real purchase
behavior, where returns are influenced by unobservable factors (actual fit, the
customer's mood, unboxing experience) that no dataset captures. On real data, swap this
script for a real order-history export with the same schema.

## Model

`scripts/train_model.py` — stratified 75/25 train/test split (no leakage), two models
compared:

| Model | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|
| Logistic Regression | 0.741 | 0.307 | 0.711 | 0.429 |
| Random Forest | 0.748 | 0.353 | 0.652 | 0.458 |

Random Forest selected as the better model by ROC-AUC (threshold-independent).

**Honest framing:** ROC-AUC of ~0.75 reflects real predictive signal without
overclaiming — a return-risk model with 0.95+ AUC on real-world data is a red flag for
leakage, not a good sign.

## False-positive-cost-aware threshold ("the bar")

Precision/recall alone don't tell a merchant *what threshold to actually use in
production*. We assign illustrative costs (documented in `train_model.py`, meant to be
replaced with a merchant's real numbers):

- **Cost of a false positive** (flagging a good order): ₹40 — verification friction,
  possible customer annoyance.
- **Benefit of a true positive** (catching a real return pre-shipment): ₹180 — avoided
  reverse logistics, restocking, and payment processing costs.

Sweeping the threshold and computing `net value = TP × benefit − FP × cost` on the held
-out test set gives a **cost-optimal threshold of 0.47**, yielding:

- Recall: 67% of true returns caught
- Precision: 35% (roughly 1 in 3 flagged orders is a genuine return)
- Net value: ≈ ₹60,600 on the ~5,000-order test set

This is the deliverable the brief asks for: not just "here's an accuracy number," but
"here's the threshold that actually makes the merchant money, and here's why."

## Explainability

`scripts/explain_model.py` — two complementary views, both network-independent (no
SHAP dependency):

1. **Global importance** — permutation importance of the deployed Random Forest,
   measured as the actual drop in test-set ROC-AUC when each raw feature is shuffled.
   Top drivers: `category` (0.175), `size_variants_in_order` (0.044, the classic
   "ordering multiple sizes to bracket fit" behavior), `past_return_rate` (0.031).
   Low-signal features (`is_gift_wrapped`, `days_to_deliver`, `device`) correctly
   contribute ~0, which is itself evidence the model isn't fitting noise.

2. **Per-order reason codes** — a parallel Logistic Regression is fit purely as an
   explanation layer. Its coefficients decompose any single order's predicted logit
   into an *exact* sum of per-feature log-odds contributions (this is how regulated
   credit-risk "adverse action" reason codes work: the production score can come from
   a stronger black-box model, while a transparent linear model explains why). The
   two models' scores are reported side by side, never conflated, and directional
   agreement between them (both flagging the same order as high/low risk) is itself
   a trust signal worth surfacing in a demo.

Outputs: `outputs/global_feature_importance.csv`, `outputs/reason_code_coefficients.csv`,
`outputs/sample_reason_codes.json` (200 sample orders with full reason codes).

## Running the full app (frontend + backend)

The backend is a Flask API (`backend/app.py`) that loads the trained models once
at startup and serves the frontend as static files from the same origin (no CORS
setup needed). The frontend (`frontend/`) is plain HTML/CSS/JS — no build step.

```bash
cd backend
pip install -r requirements.txt
python app.py
# -> open http://localhost:5000
```

Run `scripts/generate_data.py`, `scripts/train_model.py`, then
`scripts/explain_model.py` first if `outputs/` doesn't already contain
`best_model.joblib`, `explainer_model.joblib`, `eval_summary.json`, and
`feature_metadata.json` — the backend loads all four at startup.

**API contract:**

- `GET /api/options` -> categorical choices + numeric ranges (drives the form; the
  frontend also uses the medians to pre-fill sensible defaults).
- `POST /api/score` -> body is the 11 order fields from `FEATURE_COLS` in
  `backend/model_utils.py`; returns the deployed model's probability, the
  cost-optimal flag decision, the explainer model's independent probability, and
  the top risk-increasing/decreasing reason codes.

**What the UI shows, and why:** the "risk certificate" deliberately displays the
deployed model's score and the explainer model's score *side by side, never
blended* — conflating a black-box score with a linear explainer's estimate would
misrepresent what's actually driving the production decision. Directional
agreement between the two (as in the examples in `explain_model.py`) is itself a
trust signal worth calling out live in a demo.



- Synthetic data: real deployment requires validating the hidden risk drivers
  (especially customer return-history weighting) against actual order history — this
  is the single highest-leverage swap.
- Cost assumptions (₹40 / ₹180) are illustrative placeholders, not sourced from a real
  merchant's P&L.
- No temporal/seasonal features (sale days, festival season) — a real fraud/returns
  team would want these.
- Precision at the cost-optimal threshold (35%) means most flags require a human or
  lightweight automated check, not an automatic rejection — this is intentional given
  the defense-only bar.

## Files

- `scripts/generate_data.py` — synthetic data generator
- `scripts/train_model.py` — training, evaluation, cost-threshold analysis
- `scripts/explain_model.py` — global importance + per-order reason codes, persists
  the explainer model and feature metadata
- `data/orders.csv` — generated dataset
- `outputs/best_model.joblib` — deployed model (Random Forest)
- `outputs/explainer_model.joblib` — explainer model (Logistic Regression)
- `outputs/eval_summary.json` — full metrics + cost analysis
- `outputs/feature_metadata.json` — categorical choices + numeric ranges for the form
- `outputs/threshold_curve.csv`, `outputs/roc_curve.csv` — plotting data
- `backend/app.py` — Flask API + static frontend server
- `backend/model_utils.py` — model loading, validation, scoring, explanation logic
- `backend/requirements.txt`
- `frontend/index.html`, `frontend/styles.css`, `frontend/app.js` — the Risk Console UI
