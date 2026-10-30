import duckdb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs('outputs', exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Clean Master Data
# -----------------------------------------------------------------------------
print("Loading clean dataset from olist.duckdb (table: order_master)...")
con = duckdb.connect("olist.duckdb")
df = con.execute("SELECT * FROM order_master").df()
print(f"Loaded {len(df):,} clean rows with {df.shape[1]} raw columns.")

# -----------------------------------------------------------------------------
# 2. Group Rare Categories (Avoid 70+ column explosion)
# -----------------------------------------------------------------------------
# Customer State: Keep top 10 states (covers 90.5% of volume), group rest into 'other'
top_cust_states = df['customer_state'].value_counts().nlargest(10).index
df['cust_state_grouped'] = df['customer_state'].apply(lambda x: x if x in top_cust_states else 'other')

# Seller State: Keep top 5 states (covers 94.6% of sellers), group rest into 'other'
top_seller_states = df['seller_state'].value_counts().nlargest(5).index
df['seller_state_grouped'] = df['seller_state'].apply(lambda x: x if x in top_seller_states else 'other')

# Product Category: Keep top 12 categories (covers 71.3% of catalog), group rest into 'other'
top_categories = df['category_en'].value_counts().nlargest(12).index
df['category_grouped'] = df['category_en'].apply(lambda x: x if x in top_categories else 'other')

print("\n--- Categorical Grouping Summary ---")
print(f"Customer States:    10 top + 'other' = {df['cust_state_grouped'].nunique()} categories")
print(f"Seller States:      5 top + 'other'  = {df['seller_state_grouped'].nunique()} categories")
print(f"Product Categories: 12 top + 'other' = {df['category_grouped'].nunique()} categories")

# -----------------------------------------------------------------------------
# 3. Log-Transform Skewed Numerical Variables
# -----------------------------------------------------------------------------
# Using ln(1 + x) for zero-safe transformation on right-skewed variables
df['log_weight'] = np.log1p(df['weight_capped'])
df['log_distance'] = np.log1p(df['distance_km'])
df['log_price'] = np.log1p(df['total_price'])
df['log_freight'] = np.log1p(df['total_freight'])
df['log_volume'] = np.log1p(df['volume_capped'])

print("\n--- Skewness Reduction via Log Transforms ---")
skew_check = pd.DataFrame({
    'Variable': ['weight', 'distance', 'price', 'freight', 'volume'],
    'Raw Skewness': [
        df['weight_capped'].skew(), df['distance_km'].skew(), df['total_price'].skew(),
        df['total_freight'].skew(), df['volume_capped'].skew()
    ],
    'Log1p Skewness': [
        df['log_weight'].skew(), df['log_distance'].skew(), df['log_price'].skew(),
        df['log_freight'].skew(), df['log_volume'].skew()
    ]
})
print(skew_check.to_string(index=False))

# -----------------------------------------------------------------------------
# 4. Construct Feature Matrix (X) and Targets (y)
# -----------------------------------------------------------------------------
# Continuous variables to be standardized
continuous_cols = [
    'log_weight',
    'log_distance',
    'log_price',
    'log_freight',
    'log_volume',
    'freight_ratio_capped',
    'seller_handling_days',
    'estimated_days',
    'n_items',
    'n_sellers'
]

# Binary / Indicator variables (not standardized, kept on 0/1 scale)
binary_cols = ['same_state']

# Categorical dummy variables with drop_first=True to prevent multicollinearity
dummy_features = pd.get_dummies(
    df[['cust_state_grouped', 'seller_state_grouped', 'category_grouped']],
    prefix=['cust_state', 'seller_state', 'cat'],
    drop_first=True,
    dtype=float
)

print(f"\nTotal dummy columns created: {dummy_features.shape[1]}")

# Combine continuous and binary unscaled features
X_raw = pd.concat([df[continuous_cols + binary_cols], dummy_features], axis=1)
y_late = df['is_late']
y_review = df['low_review']
y_score = df['review_score']

# -----------------------------------------------------------------------------
# 5. Train / Test Split (80/20, Fixed Random Seed = 42)
# -----------------------------------------------------------------------------
# Stratify by is_late to maintain balanced 8.11% late rate in both train and test sets
X_train_raw, X_test_raw, y_train_late, y_test_late, idx_train, idx_test = train_test_split(
    X_raw, y_late, df.index, test_size=0.20, random_state=42, stratify=y_late
)

y_train_review = y_review.loc[idx_train]
y_test_review = y_review.loc[idx_test]
y_train_score = y_score.loc[idx_train]
y_test_score = y_score.loc[idx_test]

print(f"\n--- Train / Test Split (80/20, random_state=42) ---")
print(f"Training set:   {len(X_train_raw):,} rows (80.0%) | Lateness rate: {y_train_late.mean():.2%}")
print(f"Test set:       {len(X_test_raw):,} rows (20.0%) | Lateness rate: {y_test_late.mean():.2%}")

# -----------------------------------------------------------------------------
# 6. Standardize Continuous Variables (Fit on Train, Transform Test)
# -----------------------------------------------------------------------------
scaler = StandardScaler()

# Fit scaler exclusively on training continuous variables (prevents data leakage)
X_train_cont_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw[continuous_cols]),
    columns=continuous_cols,
    index=X_train_raw.index
)

X_test_cont_scaled = pd.DataFrame(
    scaler.transform(X_test_raw[continuous_cols]),
    columns=continuous_cols,
    index=X_test_raw.index
)

# Reassemble full feature matrices with scaled continuous + unscaled binary/dummies
non_cont_cols = [col for col in X_raw.columns if col not in continuous_cols]

X_train = pd.concat([X_train_cont_scaled, X_train_raw[non_cont_cols]], axis=1)
X_test = pd.concat([X_test_cont_scaled, X_test_raw[non_cont_cols]], axis=1)

print(f"\nFinal feature matrix shape: {X_train.shape[1]} features")
print(f"  - Standardized continuous features: {len(continuous_cols)}")
print(f"  - Binary routing features:          {len(binary_cols)}")
print(f"  - One-hot categorical dummies:      {dummy_features.shape[1]}")

# -----------------------------------------------------------------------------
# 7. Verification: Verify Mean = 0, Std = 1 on Train Continuous Features
# -----------------------------------------------------------------------------
scaler_check = pd.DataFrame({
    'Feature': continuous_cols,
    'Train Mean (should be ~0.0)': X_train[continuous_cols].mean().round(4),
    'Train Std (should be 1.0)': X_train[continuous_cols].std().round(4),
    'Test Mean': X_test[continuous_cols].mean().round(4),
    'Test Std': X_test[continuous_cols].std().round(4)
})
print("\n=== STANDARDIZATION CHECK TABLE ===")
print(scaler_check.to_string(index=False))

# -----------------------------------------------------------------------------
# 8. Save Preprocessed Feature Matrices for Regression Modeling
# -----------------------------------------------------------------------------
# Save feature metadata
feature_meta = pd.DataFrame({
    'feature_name': X_train.columns,
    'feature_type': ['continuous_standardized' if c in continuous_cols else 
                    ('binary' if c in binary_cols else 'dummy') for c in X_train.columns]
})
feature_meta.to_csv('outputs/model_feature_metadata.csv', index=False)

# Export train/test datasets for modeling
X_train.to_csv('outputs/X_train.csv', index=False)
X_test.to_csv('outputs/X_test.csv', index=False)
pd.DataFrame({'is_late': y_train_late, 'low_review': y_train_review, 'review_score': y_train_score}).to_csv('outputs/y_train.csv', index=False)
pd.DataFrame({'is_late': y_test_late, 'low_review': y_test_review, 'review_score': y_test_score}).to_csv('outputs/y_test.csv', index=False)

print("\nSaved preprocessed feature matrices and targets to outputs/:")
print("  - outputs/X_train.csv (76,636 rows x 38 features)")
print("  - outputs/X_test.csv  (19,159 rows x 38 features)")
print("  - outputs/y_train.csv")
print("  - outputs/y_test.csv")
print("  - outputs/model_feature_metadata.csv")
print("\nFeature preparation completed successfully!")
