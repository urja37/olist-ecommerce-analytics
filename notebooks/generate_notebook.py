import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Context
cells.append(nbf.v4.new_markdown_cell("""# Olist E-Commerce Analytics: Predictive & Root-Cause Analysis
**Author**: Senior E-Commerce Data Scientist & Operations Analyst  
**Dataset**: Brazilian E-Commerce Public Dataset by Olist (99,441 orders)  
**Core Questions Addressed**:
1. What factors predict whether an order arrives late?
2. What factors predict a low review score?
3. How much of the low-review problem is explained by lateness?
4. What actionable recommendations should Olist take to reduce lateness and low reviews?
"""))

# Setup & Libraries
cells.append(nbf.v4.new_code_cell("""import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 50)
con = duckdb.connect()
print("Environment and DuckDB initialized.")
"""))

# Load Analytical Dataset
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Pre-Engineered Master Analytical Dataset
We use the pre-built master analytical table exported by our DuckDB modeling pipeline.
Each row represents a delivered order with enriched features for items, products, sellers, payments, geolocations, and reviews.
"""))

cells.append(nbf.v4.new_code_cell("""df = con.execute("SELECT * FROM 'outputs/olist_analytical_dataset.parquet'").df()
print(f"Loaded master dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head(3)
"""))

# Question 1: Late Delivery Prediction
cells.append(nbf.v4.new_markdown_cell("""## 2. Question 1: What Factors Predict Whether an Order Arrives Late?
- **Target**: `is_late` (`order_delivered_customer_date > order_estimated_delivery_date`).
- **Baseline Lateness Rate**: 8.11% (7,826 late orders out of 96,470 delivered orders).
"""))

cells.append(nbf.v4.new_code_cell("""odds_q1 = pd.read_csv('outputs/q1_odds_ratios.csv')
feat_q1 = pd.read_csv('outputs/q1_feature_importance.csv')
routes = pd.read_csv('outputs/q1_top_risk_routes.csv')
cats = pd.read_csv('outputs/q1_top_risk_categories.csv')
bottlenecks = pd.read_csv('outputs/q1_bottleneck_breakdown.csv')

print("=== TOP 8 RISK FACTORS FOR LATE DELIVERY (ODDS RATIOS) ===")
print(odds_q1.head(8))

print("\n=== TOP 8 TREE FEATURE IMPORTANCES (ROC-AUC DECREASE) ===")
print(feat_q1.head(8))
"""))

cells.append(nbf.v4.new_code_cell("""from IPython.display import Image
Image(filename='outputs/q1_feature_importance.png')
"""))

cells.append(nbf.v4.new_code_cell("""print("=== TOP 5 HIGHEST-RISK SHIPPING ROUTES ===")
print(routes.head(5))
Image(filename='outputs/q1_high_risk_routes.png')
"""))

cells.append(nbf.v4.new_markdown_cell("""### Fulfillment Bottleneck Decomposition
Where does the delay actually occur? Approval, Seller Handling, or Carrier Transit?
"""))

cells.append(nbf.v4.new_code_cell("""print(bottlenecks)
Image(filename='outputs/q1_bottleneck_waterfall.png')
"""))

# Question 2: Low Review Score Prediction
cells.append(nbf.v4.new_markdown_cell("""## 3. Question 2: What Factors Predict a Low Review Score?
- **Target**: `is_low_review` (`review_score <= 2`).
- **Baseline Low Review Rate**: 12.85% (12,310 orders with score <= 2).
"""))

cells.append(nbf.v4.new_code_cell("""odds_q2 = pd.read_csv('outputs/q2_odds_ratios.csv')
feat_q2 = pd.read_csv('outputs/q2_feature_importance.csv')

print("=== TOP 8 PREDICTORS OF LOW REVIEW (ODDS RATIOS) ===")
print(odds_q2.head(8))

print("\n=== TOP 8 TREE FEATURE IMPORTANCES ===")
print(feat_q2.head(8))
Image(filename='outputs/q2_feature_importance.png')
"""))

cells.append(nbf.v4.new_markdown_cell("""### The Threshold Effect: Delivery Delay Days vs. Review Score
When an order arrives even 1 day past the promised delivery date, the low review rate jumps from 9.2% to 35%+.
"""))

cells.append(nbf.v4.new_code_cell("""Image(filename='outputs/q2_delay_vs_review_score.png')
"""))

# Question 3: Lateness Contribution & Attributable Risk
cells.append(nbf.v4.new_markdown_cell("""## 4. Question 3: How Much of the Low-Review Problem is Explained by Lateness?
- **Empirical Low Review Rate**:
  - On-Time Orders: **9.26%**
  - Late Orders: **54.12%** (Relative Risk: **5.84x**, Odds Ratio: **11.56**)
- **Population Attributable Risk (PAR %)**: **27.92%**
- **Model Deviance Explained by Delivery Lateness**: **61.46%**
"""))

cells.append(nbf.v4.new_code_cell("""q3_metrics = pd.read_csv('outputs/q3_attributable_risk_metrics.csv')
print(q3_metrics)
Image(filename='outputs/q3_attributable_risk_waterfall.png')
"""))

cells.append(nbf.v4.new_markdown_cell("""### Review Comments Text Analysis
Comparison of customer complaint topics between Late Low Reviews vs. On-Time Low Reviews.
"""))

cells.append(nbf.v4.new_code_cell("""Image(filename='outputs/q3_review_topic_breakdown.png')
"""))

# Question 4: Actionable Recommendations
cells.append(nbf.v4.new_markdown_cell("""## 5. Question 4: Actionable Operational Recommendations
Prioritized strategies across Seller Management, Logistics SLA Optimization, and Platform Experience.
"""))

cells.append(nbf.v4.new_code_cell("""Image(filename='outputs/q4_recommendations_matrix.png')
"""))

nb['cells'] = cells

with open('notebooks/olist_end_to_end_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/olist_end_to_end_analysis.ipynb")
