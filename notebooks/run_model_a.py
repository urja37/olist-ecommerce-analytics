import duckdb
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
import os

os.makedirs('outputs', exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Clean Data & Build Feature Matrix
# -----------------------------------------------------------------------------
con = duckdb.connect("olist.duckdb")
df = con.execute("SELECT * FROM order_master").df()

# Group high-cardinality categories
top_cust_states = df['customer_state'].value_counts().nlargest(10).index
df['cust_state_grouped'] = df['customer_state'].apply(lambda x: x if x in top_cust_states else 'other')

top_categories = df['category_en'].value_counts().nlargest(12).index
df['category_grouped'] = df['category_en'].apply(lambda x: x if x in top_categories else 'other')

# Log-transforms for skewed predictors
df['log_weight'] = np.log1p(df['weight_capped'])
df['log_distance'] = np.log1p(df['distance_km'])
df['log_volume'] = np.log1p(df['volume_capped'])

# Continuous predictors
continuous_cols = [
    'log_distance',
    'log_weight',
    'log_volume',
    'n_items',
    'n_sellers',
    'freight_ratio_capped',
    'seller_handling_days',
    'estimated_days',
    'purchase_month'
]

# One-hot encoding for categorical variables (drop_first=True to avoid collinearity)
dummy_cols = ['cust_state_grouped', 'category_grouped']
dummies = pd.get_dummies(df[dummy_cols], prefix=['cust_state', 'cat'], drop_first=True, dtype=float)

# Feature matrix X and target y
X_all = pd.concat([df[continuous_cols + ['same_state']], dummies], axis=1)
y_all = df['is_late']

# 80/20 train/test split with fixed random seed
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

# Standardize continuous predictors (fit on train, transform test)
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# Add intercept constant
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# -----------------------------------------------------------------------------
# 2. Fit Logistic Regression (Model A)
# -----------------------------------------------------------------------------
print("Fitting Logistic Regression Model A (is_late)...")
model_a = sm.Logit(y_train, X_train_const).fit()
summary_str = str(model_a.summary())
print(summary_str)

# Save summary text
with open('outputs/model_a_summary.txt', 'w') as f:
    f.write(summary_str)

# -----------------------------------------------------------------------------
# 3. Calculate Odds Ratios & 95% Confidence Intervals
# -----------------------------------------------------------------------------
conf = model_a.conf_int()
odds_df = pd.DataFrame({
    'Feature': model_a.params.index,
    'Coefficient (Beta)': model_a.params.values,
    'Std_Error': model_a.bse.values,
    'z_stat': model_a.tvalues.values,
    'p_value': model_a.pvalues.values,
    'Odds_Ratio': np.exp(model_a.params.values),
    'CI_Lower_95': np.exp(conf[0].values),
    'CI_Upper_95': np.exp(conf[1].values)
}).sort_values('Odds_Ratio', ascending=False)

# Exclude intercept for ranking
odds_ranking = odds_df[odds_df['Feature'] != 'const'].copy()
odds_ranking.to_csv('outputs/model_a_odds_ratios.csv', index=False)

# -----------------------------------------------------------------------------
# 4. Out-of-Sample Test Set Evaluation
# -----------------------------------------------------------------------------
y_test_pred_prob = model_a.predict(X_test_const)
test_auc = roc_auc_score(y_test, y_test_pred_prob)
test_pr_auc = average_precision_score(y_test, y_test_pred_prob)

print(f"\n--- Model A Out-of-Sample Performance on Test Set ---")
print(f"Test ROC-AUC:        {test_auc:.4f}")
print(f"Test PR-AUC:         {test_pr_auc:.4f}")
print(f"Pseudo R-squared:    {model_a.prsquared:.4f}")
print(f"Log-Likelihood:      {model_a.llf:.2f} (Null: {model_a.llnull:.2f})")

print("\n=== TOP 10 RISK FACTORS FOR LATE DELIVERY (HIGHEST ODDS RATIOS) ===")
print(odds_ranking.head(10)[['Feature', 'Coefficient (Beta)', 'Odds_Ratio', 'p_value', 'CI_Lower_95', 'CI_Upper_95']].to_string(index=False))

print("\n=== TOP 5 PROTECTIVE FACTORS AGAINST LATE DELIVERY (LOWEST ODDS RATIOS) ===")
print(odds_ranking.tail(5)[['Feature', 'Coefficient (Beta)', 'Odds_Ratio', 'p_value', 'CI_Lower_95', 'CI_Upper_95']].to_string(index=False))
