# Dynamic Pricing Engine — ML-Powered Price Optimization

> **CASE STUDY** · Machine Learning · Price Optimisation · Python · XGBoost

---

## The Problem

Pricing decisions in retail are still mostly intuitive. A buyer looks at competitor prices, checks last month's margin report, and adjusts manually. This works until you have 50 products, 3 competitors, and seasonal demand curves that shift every quarter.

The result: most products are either priced too low (leaving margin on the table) or too high (suppressing volume that would have been profitable). Both are expensive mistakes.

This project builds an ML-powered pricing engine that answers one question with data:

> *For each product, what price maximises margin — and what's the expected impact?*

---

## Architecture

The engine has three stages:

```
Data (sales + competitor prices)
        ↓
 Feature Engineering
 (price ratios, elasticity proxies, lag demand, seasonality)
        ↓
  XGBoost Demand Model
  (predicts units sold at any given price)
        ↓
  Price Optimiser (scipy.optimize)
  (finds the price that maximises margin OR revenue)
        ↓
 Recommendations CSV + Interactive Dashboard
```

---

## Stage 1 — Feature Engineering

20 features engineered from raw transaction data:

| Feature Group | Features | Signal |
|---|---|---|
| **Price positioning** | price_vs_base, price_vs_avg_comp, price_vs_cheapest_comp | How do we compare to baseline and market? |
| **Competitor context** | avg_competitor_price, comp_price_spread | How tight is the competitive range? |
| **Temporal** | day_of_week, month, quarter, is_weekend | Seasonality and day-of-week effects |
| **Demand momentum** | lag_7d, lag_14d, lag_28d, rolling_14d_avg | Where is demand trending? |
| **Product context** | base_price, cost, base_daily_demand, category | Product-level baseline characteristics |

---

## Stage 2 — XGBoost Demand Model

A gradient boosted tree model predicts daily demand as a function of price and context.

**Model configuration:**
```python
XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
)
```

**Results on holdout test set (20%):**

| Metric | Value |
|--------|-------|
| R² | **0.9483** |
| MAE | 3.39 units |
| RMSE | 4.73 units |
| MAPE | 11.5% |

**Top drivers of demand (feature importance):**
1. `demand_rolling_14d_avg` — demand momentum matters most
2. `demand_lag_7d` — recent history is predictive
3. `price_vs_avg_comp` — market position drives purchasing decisions
4. `price_vs_base` — deviation from expected price signals value

---

## Stage 3 — Price Optimisation

For each product, `scipy.optimize.minimize_scalar` searches within ±30% of the base price to find the price that maximises the chosen objective:

**Margin objective:**
```
maximise: (price − cost) × predicted_demand(price)
```

**Revenue objective:**
```
maximise: price × predicted_demand(price)
```

The trade-off is intentional and visible in the results — maximising margin typically means raising prices and accepting lower volume; maximising revenue means lowering prices to grow volume.

---

## Results

### Maximise Margin
| Metric | Value |
|--------|-------|
| Products to raise | 42 / 50 |
| Products to lower | 6 / 50 |
| Average margin uplift | **+43.6%** |
| Maximum single-product uplift | **+102.1%** (Electronics Item 07) |

### Maximise Revenue
| Metric | Value |
|--------|-------|
| Products to lower | 46 / 50 |
| Products to raise | 2 / 50 |
| Average revenue uplift | **+13.9%** |
| Maximum single-product uplift | **+36.8%** |

**Key insight:** The margin and revenue objectives recommend nearly opposite actions for most products. This is the pricing dilemma made explicit — raising prices improves margin per unit but reduces volume. The right choice depends on the business's current strategy: are you optimising for profitability or market share? This engine makes both options transparent and data-driven.

---

## Price Elasticity by Category

| Category | Elasticity | Interpretation |
|----------|-----------|----------------|
| Electronics | −2.1 | Highly price-sensitive. 10% price increase → 21% demand drop |
| Sports | −1.8 | Moderate sensitivity. Brand loyalty provides some buffer |
| Apparel | −1.6 | Seasonal effects dominate over price |
| Beauty | −1.4 | Low sensitivity. Loyalty and routine purchases |
| Home & Garden | −1.2 | Need-based. Least sensitive to price changes |

---

## Stack

```
Python · XGBoost · scikit-learn · scipy.optimize · pandas · numpy
```

---

## How to Run

```bash
git clone https://github.com/klaricracia/dynamic-pricing-model.git
cd dynamic-pricing-model

pip install xgboost scikit-learn scipy pandas numpy

# Optimise for margin (default)
python pricing_engine.py

# Optimise for revenue
python pricing_engine.py --objective revenue

# Run both and compare
python pricing_engine.py --objective both
```

Output:
- `output/pricing_recommendations.csv` — full recommendations table
- `output/model_metrics.json` — model evaluation metrics + feature importance
- `dashboard/index.html` — interactive results dashboard (open in browser)

---

## How to Extend This

- **Live competitor scraping:** Connect the data pipeline to a real web scraper (see the [Retail Intelligence Scraper](https://github.com/klaricracia/retail-customer-rfm-analysis) project) to feed live competitor prices
- **Segment-aware pricing:** Integrate with the [RFM Segmentation](https://github.com/klaricracia/retail-customer-rfm-analysis) project to price differently by customer segment
- **Alerting:** Wire the KPI Monitoring Bot to trigger when actual demand deviates significantly from model predictions
- **A/B testing framework:** Add logic to test recommended prices on a subset of customers before full rollout

---

*Built by [Klarissa Artavia](https://www.linkedin.com/in/klariartavia/) · Data & AI Strategy*
*GitHub: [github.com/klaricracia](https://github.com/klaricracia)*
