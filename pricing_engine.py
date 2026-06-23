"""
Dynamic Pricing Engine — ML-Powered Price Optimization
=======================================================
Author: Klarissa Artavia — Data & AI Strategy
GitHub: https://github.com/klaricracia
LinkedIn: https://www.linkedin.com/in/klariartavia/

Pipeline:
  1. Feature Engineering  — price ratios, elasticity proxies, lag demand, seasonality
  2. Demand Model         — XGBoost regressor predicts demand at any price point
  3. Price Optimizer      — scipy.optimize finds the price that maximises margin OR revenue
  4. Recommendations      — output per-product recommended price + expected impact

Usage:
    python pricing_engine.py                     # optimise for margin (default)
    python pricing_engine.py --objective revenue # optimise for revenue
    python pricing_engine.py --objective both    # run both and compare
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize_scalar
import json, os, argparse, warnings
warnings.filterwarnings('ignore')

os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DYNAMIC PRICING ENGINE")
print("="*60)

print("\n[1/5] Loading data...")
sales    = pd.read_csv("data/sales_history.csv", parse_dates=["date"])
comps    = pd.read_csv("data/competitor_prices.csv", parse_dates=["date"])
products = pd.read_csv("data/products.csv")

df = sales.drop(columns=["cost"]).merge(comps, on=["date","product_id"]).merge(
    products[["product_id","category","base_price","cost","base_daily_demand"]],
    on="product_id"
)
print(f"  {len(df):,} records · {df['product_id'].nunique()} products · {df['date'].nunique()} days")

# ── Feature Engineering ────────────────────────────────────────────────────────
print("\n[2/5] Engineering features...")

# Price positioning features
df["price_vs_base"]         = df["price"] / df["base_price"]
df["price_vs_avg_comp"]     = df["price"] / df["avg_competitor_price"]
df["price_vs_cheapest_comp"]= df["price"] / df[["comp_a_price","comp_b_price","comp_c_price"]].min(axis=1)
df["comp_price_spread"]     = (df[["comp_a_price","comp_b_price","comp_c_price"]].max(axis=1) -
                                df[["comp_a_price","comp_b_price","comp_c_price"]].min(axis=1)) / df["base_price"]
df["margin_pct"]            = (df["price"] - df["cost"]) / df["price"]

# Temporal features
df["day_of_week"]  = df["date"].dt.dayofweek
df["month"]        = df["date"].dt.month
df["quarter"]      = df["date"].dt.quarter
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# Lag demand features (demand momentum)
df = df.sort_values(["product_id","date"])
for lag in [7, 14, 28]:
    df[f"demand_lag_{lag}d"] = df.groupby("product_id")["demand"].shift(lag)
df["demand_rolling_14d_avg"] = df.groupby("product_id")["demand"].transform(
    lambda x: x.shift(1).rolling(14, min_periods=3).mean()
)

# Category encoding
le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["category"])

df = df.dropna()
print(f"  {len(df):,} records after feature engineering")

FEATURES = [
    "price", "price_vs_base", "price_vs_avg_comp", "price_vs_cheapest_comp",
    "comp_price_spread", "margin_pct", "avg_competitor_price",
    "day_of_week", "month", "quarter", "is_weekend", "week_of_year",
    "base_daily_demand", "demand_lag_7d", "demand_lag_14d", "demand_lag_28d",
    "demand_rolling_14d_avg", "category_enc", "base_price", "cost",
]

X = df[FEATURES]
y = df["demand"]

# ── Train XGBoost Demand Model ─────────────────────────────────────────────────
print("\n[3/5] Training XGBoost demand model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluation
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
mape   = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

print(f"  MAE:  {mae:.2f} units")
print(f"  RMSE: {rmse:.2f} units")
print(f"  R²:   {r2:.4f}")
print(f"  MAPE: {mape:.1f}%")

# Feature importance
fi = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# Save model metrics
metrics = {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4), "MAPE": round(mape,2)}
with open("output/model_metrics.json","w") as f:
    json.dump({"model_metrics": metrics, "top_features": fi.head(8).to_dict("records")}, f, indent=2)

# ── Price Optimizer ────────────────────────────────────────────────────────────
print("\n[4/5] Optimising prices...")

def predict_demand(price, row, feature_cols):
    """Predict demand for a given price using the trained model."""
    sample = row.copy()
    sample["price"]            = price
    sample["price_vs_base"]    = price / sample["base_price"]
    sample["price_vs_avg_comp"]= price / sample["avg_competitor_price"]
    sample["price_vs_cheapest_comp"] = price / min(
        sample["comp_a_price"], sample["comp_b_price"], sample["comp_c_price"]
    )
    sample["margin_pct"] = (price - sample["cost"]) / price
    return max(0, model.predict(pd.DataFrame([sample[feature_cols]]))[0])

def optimise_price(row, objective="margin"):
    """Find the price that maximises margin or revenue for a single product."""
    base    = row["base_price"]
    cost    = row["cost"]
    lo, hi  = base * 0.70, base * 1.40   # search bounds ±30% of base price

    def neg_objective(price):
        demand = predict_demand(price, row, FEATURES)
        if objective == "margin":
            return -((price - cost) * demand)
        else:
            return -(price * demand)

    result = minimize_scalar(neg_objective, bounds=(lo, hi), method="bounded")
    opt_price  = round(result.x, 2)
    opt_demand = predict_demand(opt_price, row, FEATURES)

    curr_demand = predict_demand(row["price"], row, FEATURES)
    curr_margin = (row["price"] - cost) * curr_demand
    opt_margin  = (opt_price - cost) * opt_demand
    curr_revenue = row["price"] * curr_demand
    opt_revenue  = opt_price * opt_demand

    return {
        "current_price":    round(row["price"], 2),
        "optimal_price":    opt_price,
        "price_change_pct": round((opt_price - row["price"]) / row["price"] * 100, 1),
        "current_demand":   round(curr_demand, 1),
        "optimal_demand":   round(opt_demand, 1),
        "current_margin":   round(curr_margin, 2),
        "optimal_margin":   round(opt_margin, 2),
        "margin_uplift_pct":round((opt_margin - curr_margin) / (curr_margin + 1e-6) * 100, 1),
        "current_revenue":  round(curr_revenue, 2),
        "optimal_revenue":  round(opt_revenue, 2),
        "revenue_uplift_pct":round((opt_revenue - curr_revenue) / (curr_revenue + 1e-6) * 100, 1),
    }

# Use the most recent day's data as the "current state" for each product
latest = df.sort_values("date").groupby("product_id").last().reset_index()

parser = argparse.ArgumentParser()
parser.add_argument("--objective", choices=["margin","revenue","both"], default="margin")
args = parser.parse_args()

objectives = ["margin","revenue"] if args.objective == "both" else [args.objective]
all_recs = []

for obj in objectives:
    recs = []
    for _, row in latest.iterrows():
        result = optimise_price(row, objective=obj)
        rec = {
            "product_id":   int(row["product_id"]),
            "product_name": products.loc[products["product_id"]==row["product_id"],"product_name"].values[0],
            "category":     row["category"],
            "objective":    obj,
            **result,
            "avg_competitor_price": round(row["avg_competitor_price"], 2),
            "action": ("RAISE" if result["price_change_pct"] > 2
                       else "LOWER" if result["price_change_pct"] < -2
                       else "HOLD"),
        }
        recs.append(rec)
    all_recs.extend(recs)

recs_df = pd.DataFrame(all_recs)
recs_df.to_csv("output/pricing_recommendations.csv", index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n[5/5] Results summary...")
print("="*60)

for obj in objectives:
    sub = recs_df[recs_df["objective"] == obj]
    print(f"\n  Objective: MAXIMISE {obj.upper()}")
    print(f"  {'RAISE':>6} prices: {len(sub[sub['action']=='RAISE']):>3} products")
    print(f"  {'LOWER':>6} prices: {len(sub[sub['action']=='LOWER']):>3} products")
    print(f"  {'HOLD':>6}  prices: {len(sub[sub['action']=='HOLD']):>3} products")
    key = "margin" if obj == "margin" else "revenue"
    print(f"  Avg {key} uplift: +{sub[f'{key}_uplift_pct'].mean():.1f}%")
    print(f"  Max {key} uplift: +{sub[f'{key}_uplift_pct'].max():.1f}%")

print(f"\n  Recommendations saved → output/pricing_recommendations.csv")
print(f"  Model metrics saved   → output/model_metrics.json")
print("="*60 + "\n")
