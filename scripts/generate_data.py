"""
generate_data.py
Generates a synthetic e-commerce order dataset for a RETURN-RISK SCORER.

Each row = one order. Target column `returned` = 1 if the order was returned.

We deliberately build the "true" return probability from a hidden formula
based on realistic drivers, then add noise -- so no single feature perfectly
predicts the label (like real life), but the model should still learn the
real signal.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ORDERS = 20000

rng = np.random.default_rng(RNG_SEED)

# ---- Categorical feature pools -------------------------------------------------
categories = ["Apparel", "Footwear", "Electronics", "Home", "Beauty", "Books", "Toys"]
# Apparel/Footwear have inherently higher return rates (sizing/fit issues) -- realistic.
category_base_risk = {
    "Apparel": 0.055,
    "Footwear": 0.05,
    "Electronics": 0.01,
    "Home": 0.015,
    "Beauty": 0.02,
    "Books": 0.005,
    "Toys": 0.015,
}

payment_methods = ["COD", "UPI", "Card", "NetBanking", "Wallet"]
# Cash-on-delivery orders in Indian BFSI/e-comm context tend to have higher return
# rates (lower commitment at purchase time).
payment_risk_boost = {
    "COD": 0.05,
    "UPI": 0.0,
    "Card": 0.0,
    "NetBanking": 0.0,
    "Wallet": 0.0,
}

devices = ["Mobile", "Desktop", "Tablet"]

# ---- Generate raw features ------------------------------------------------------
category = rng.choice(categories, size=N_ORDERS, p=[0.30, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05])
payment_method = rng.choice(payment_methods, size=N_ORDERS, p=[0.35, 0.30, 0.20, 0.10, 0.05])
device = rng.choice(devices, size=N_ORDERS, p=[0.65, 0.25, 0.10])

price = np.round(np.exp(rng.normal(6.2, 0.9, size=N_ORDERS)), 2)  # skewed, ~ INR 100-15000
price = np.clip(price, 99, 25000)

discount_pct = np.clip(rng.beta(2, 6, size=N_ORDERS) * 100, 0, 90)  # most orders low discount

# Customer history: how many past orders, and past return rate (0 for new customers)
past_orders = rng.poisson(4, size=N_ORDERS)
past_orders = np.clip(past_orders, 0, 60)

# Customers with more past orders have a somewhat stable personal return-rate "trait"
customer_trait_return_rate = np.clip(rng.beta(1.5, 12, size=N_ORDERS), 0, 0.6)
# New customers (0 past orders) -> trait is noisier / less reliable signal
past_return_rate = np.where(
    past_orders == 0,
    0.0,
    np.clip(customer_trait_return_rate + rng.normal(0, 0.03, size=N_ORDERS), 0, 1),
)

days_to_deliver = np.clip(rng.normal(4, 2, size=N_ORDERS).round().astype(int), 1, 15)

is_gift_wrapped = rng.binomial(1, 0.08, size=N_ORDERS)
size_variants_in_order = rng.integers(1, 4, size=N_ORDERS)  # apparel/footwear: ordering multiple sizes
review_score_of_product = np.clip(rng.normal(4.1, 0.6, size=N_ORDERS), 1, 5).round(1)

# Sizing hesitation: multiple sizes ordered strongly predicts return for
# Apparel/Footwear specifically (classic "bracketing" behavior)
is_bracketed_size_order = ((category == "Apparel") | (category == "Footwear")) & (size_variants_in_order > 1)

# ---- Hidden "true" risk formula (logit space) -----------------------------------
logit = np.full(N_ORDERS, -3.2)  # base rate anchor (~low overall return rate)

logit += np.array([category_base_risk[c] for c in category]) * 10
logit += np.array([payment_risk_boost[p] for p in payment_method]) * 6
logit += (discount_pct / 100) * 0.8          # heavy discounts -> slightly more "impulse" returns
logit += (price / 25000) * 1.2                # pricier items -> more scrutiny/returns
logit += past_return_rate * 4.5               # strongest signal: customer's own history
logit += is_bracketed_size_order.astype(float) * 1.6
logit -= (review_score_of_product - 3) * 0.35  # well-reviewed products returned less
logit += (days_to_deliver > 7).astype(float) * 0.3  # slow delivery -> mild dissatisfaction
logit += rng.normal(0, 0.6, size=N_ORDERS)    # irreducible noise

prob_return = 1 / (1 + np.exp(-logit))
returned = rng.binomial(1, prob_return)

df = pd.DataFrame({
    "order_id": [f"ORD{100000+i}" for i in range(N_ORDERS)],
    "category": category,
    "price": price,
    "discount_pct": discount_pct.round(1),
    "payment_method": payment_method,
    "device": device,
    "past_orders": past_orders,
    "past_return_rate": past_return_rate.round(3),
    "days_to_deliver": days_to_deliver,
    "is_gift_wrapped": is_gift_wrapped,
    "size_variants_in_order": size_variants_in_order,
    "review_score_of_product": review_score_of_product,
    "returned": returned,
})

out_path = "/home/claude/return_risk_scorer/data/orders.csv"
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print(f"Overall return rate: {df['returned'].mean():.3%}")
print(df.groupby("category")["returned"].mean().sort_values(ascending=False))
