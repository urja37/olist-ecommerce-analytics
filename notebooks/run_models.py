import duckdb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
con = duckdb.connect()

print("Querying master analytical dataset...", flush=True)
master_sql = """
WITH geo_clean AS (
    SELECT 
        geolocation_zip_code_prefix as zip_prefix,
        MEDIAN(geolocation_lat) as lat,
        MEDIAN(geolocation_lng) as lng
    FROM read_csv_auto('data/olist_geolocation_dataset.csv')
    WHERE geolocation_lat BETWEEN -34.0 AND 5.5
      AND geolocation_lng BETWEEN -74.0 AND -34.0
    GROUP BY geolocation_zip_code_prefix
),
order_items_agg AS (
    SELECT 
        oi.order_id,
        COUNT(oi.order_item_id) as item_count,
        COUNT(DISTINCT oi.product_id) as unique_products,
        COUNT(DISTINCT oi.seller_id) as unique_sellers,
        ANY_VALUE(oi.seller_id) as primary_seller_id,
        SUM(oi.price) as total_price,
        SUM(oi.freight_value) as total_freight,
        MAX(oi.shipping_limit_date) as max_shipping_limit_date,
        SUM(COALESCE(p.product_weight_g, 0)) as total_weight_g,
        SUM(COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, 0)) as total_volume_cm3,
        ANY_VALUE(COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')) as primary_category,
        AVG(COALESCE(p.product_photos_qty, 1)) as avg_photos_qty,
        AVG(COALESCE(p.product_description_lenght, 0)) as avg_desc_length
    FROM read_csv_auto('data/olist_order_items_dataset.csv') oi
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p ON oi.product_id = p.product_id
    LEFT JOIN read_csv_auto('data/product_category_name_translation.csv') t ON p.product_category_name = t.product_category_name
    GROUP BY oi.order_id
),
payments_agg AS (
    SELECT 
        order_id,
        SUM(payment_value) as total_payment_value,
        MAX(payment_installments) as max_installments,
        ANY_VALUE(payment_type) as primary_payment_type
    FROM read_csv_auto('data/olist_order_payments_dataset.csv')
    GROUP BY order_id
),
reviews_agg AS (
    SELECT 
        order_id,
        MIN(review_score) as min_review_score,
        AVG(review_score) as avg_review_score
    FROM read_csv_auto('data/olist_order_reviews_dataset.csv')
    GROUP BY order_id
)
SELECT 
    o.order_id,
    o.customer_id,
    c.customer_state,
    s.seller_state,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    
    -- Targets
    CASE 
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1
        ELSE 0
    END as is_late,
    
    r.min_review_score as review_score,
    CASE 
        WHEN r.min_review_score <= 2 THEN 1 
        WHEN r.min_review_score IS NOT NULL THEN 0 
        ELSE NULL 
    END as is_low_review,

    -- Features
    i.item_count,
    i.unique_products,
    i.unique_sellers,
    i.primary_seller_id,
    i.total_price,
    i.total_freight,
    i.total_freight / NULLIF(i.total_price, 0) as freight_to_price_ratio,
    i.total_weight_g,
    i.total_volume_cm3,
    i.primary_category,
    i.avg_photos_qty,
    i.avg_desc_length,
    i.max_shipping_limit_date,

    p.total_payment_value,
    p.max_installments,
    p.primary_payment_type,

    c_geo.lat as customer_lat,
    c_geo.lng as customer_lng,
    s_geo.lat as seller_lat,
    s_geo.lng as seller_lng

FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_customers_dataset.csv') c ON o.customer_id = c.customer_id
JOIN order_items_agg i ON o.order_id = i.order_id
LEFT JOIN read_csv_auto('data/olist_sellers_dataset.csv') s ON i.primary_seller_id = s.seller_id
LEFT JOIN payments_agg p ON o.order_id = p.order_id
LEFT JOIN reviews_agg r ON o.order_id = r.order_id
LEFT JOIN geo_clean c_geo ON c.customer_zip_code_prefix = c_geo.zip_prefix
LEFT JOIN geo_clean s_geo ON s.seller_zip_code_prefix = s_geo.zip_prefix
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
"""

df = con.execute(master_sql).df()

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6367.0 * c

df['distance_km'] = haversine_np(df['customer_lng'], df['customer_lat'], df['seller_lng'], df['seller_lat'])
df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['max_shipping_limit_date'] = pd.to_datetime(df['max_shipping_limit_date'])

# Time features
df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 86400.0
df['carrier_transit_days'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.total_seconds() / 86400.0
df['seller_handling_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / 86400.0
df['seller_dispatch_delay_days'] = (df['order_delivered_carrier_date'] - df['max_shipping_limit_date']).dt.total_seconds() / 86400.0
df['is_seller_late_dispatch'] = (df['seller_dispatch_delay_days'] > 0).astype(int)
df['approval_delay_hours'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600.0

df['purchase_dow'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['is_interstate'] = (df['seller_state'] != df['customer_state']).astype(int)

state_to_region = {
    'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
    'RS': 'South', 'SC': 'South', 'PR': 'South',
    'DF': 'Center-West', 'GO': 'Center-West', 'MT': 'Center-West', 'MS': 'Center-West',
    'BA': 'Northeast', 'PE': 'Northeast', 'CE': 'Northeast', 'MA': 'Northeast',
    'PB': 'Northeast', 'RN': 'Northeast', 'AL': 'Northeast', 'SE': 'Northeast', 'PI': 'Northeast',
    'PA': 'North', 'AM': 'North', 'RO': 'North', 'TO': 'North', 'AC': 'North', 'AP': 'North', 'RR': 'North'
}
df['customer_region'] = df['customer_state'].map(state_to_region).fillna('Other')
df['seller_region'] = df['seller_state'].map(state_to_region).fillna('Other')

# Historical seller metrics
seller_stats = df.groupby('primary_seller_id').agg(
    seller_orders=('order_id', 'count'),
    seller_late_dispatch_rate=('is_seller_late_dispatch', 'mean'),
    seller_avg_review=('review_score', 'mean')
).reset_index()

df = df.merge(seller_stats, on='primary_seller_id', how='left')
df['seller_late_dispatch_rate'] = df['seller_late_dispatch_rate'].fillna(df['seller_late_dispatch_rate'].median())
df['seller_avg_review'] = df['seller_avg_review'].fillna(df['seller_avg_review'].median())

num_cols = ['total_price', 'total_freight', 'freight_to_price_ratio', 'total_weight_g',
            'total_volume_cm3', 'avg_photos_qty', 'avg_desc_length', 'max_installments',
            'seller_handling_days', 'seller_dispatch_delay_days', 'approval_delay_hours',
            'distance_km', 'estimated_delivery_days', 'carrier_transit_days', 'delay_days',
            'actual_delivery_days']

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# Save clean analytical dataset using DuckDB native parquet export
con.register('df_export', df)
con.execute("COPY df_export TO 'outputs/olist_analytical_dataset.parquet' (FORMAT PARQUET)")
print("Saved analytical dataset to outputs/olist_analytical_dataset.parquet", flush=True)

# =========================================================================
# QUESTION 1: PREDICTING LATE DELIVERY
# =========================================================================
print("\n" + "="*60, flush=True)
print("QUESTION 1: PREDICTING LATE DELIVERY (is_late)", flush=True)
print("="*60, flush=True)

q1_features = [
    'distance_km', 'total_freight', 'freight_to_price_ratio',
    'total_weight_g', 'total_volume_cm3', 'item_count',
    'seller_handling_days', 'is_seller_late_dispatch', 'seller_dispatch_delay_days',
    'approval_delay_hours', 'estimated_delivery_days',
    'is_interstate', 'purchase_dow', 'purchase_month',
    'seller_orders', 'seller_late_dispatch_rate'
]

q1_df = pd.get_dummies(df[q1_features + ['customer_region', 'customer_state', 'is_late']], 
                       columns=['customer_region', 'customer_state'], drop_first=True)

X_q1 = q1_df.drop(columns=['is_late'])
y_q1 = q1_df['is_late']

X_q1_train, X_q1_test, y_q1_train, y_q1_test = train_test_split(X_q1, y_q1, test_size=0.2, random_state=42, stratify=y_q1)

# Standardized Logistic Regression
scaler_q1 = StandardScaler()
X_q1_train_sc = scaler_q1.fit_transform(X_q1_train)
X_q1_test_sc = scaler_q1.transform(X_q1_test)

log_reg_q1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg_q1.fit(X_q1_train_sc, y_q1_train)
y_q1_prob_lr = log_reg_q1.predict_proba(X_q1_test_sc)[:, 1]

auc_lr_q1 = roc_auc_score(y_q1_test, y_q1_prob_lr)
ap_lr_q1 = average_precision_score(y_q1_test, y_q1_prob_lr)

odds_ratios_q1 = pd.DataFrame({
    'Feature': X_q1.columns,
    'Coefficient': log_reg_q1.coef_[0],
    'Odds_Ratio': np.exp(log_reg_q1.coef_[0])
}).sort_values('Odds_Ratio', ascending=False)
odds_ratios_q1.to_csv('outputs/q1_odds_ratios.csv', index=False)

print(f"Logistic Regression Q1 -> ROC-AUC: {auc_lr_q1:.4f}, PR-AUC: {ap_lr_q1:.4f}", flush=True)

# HistGradientBoosting Classifier
hgb_q1 = HistGradientBoostingClassifier(random_state=42, class_weight='balanced', max_iter=100)
hgb_q1.fit(X_q1_train, y_q1_train)
y_q1_prob_hgb = hgb_q1.predict_proba(X_q1_test)[:, 1]

auc_hgb_q1 = roc_auc_score(y_q1_test, y_q1_prob_hgb)
ap_hgb_q1 = average_precision_score(y_q1_test, y_q1_prob_hgb)
print(f"HistGradientBoosting Q1 -> ROC-AUC: {auc_hgb_q1:.4f}, PR-AUC: {ap_hgb_q1:.4f}", flush=True)

# Permutation importance on a 4,000 sample test set for high speed and precision
sample_idx = np.random.RandomState(42).choice(len(X_q1_test), size=4000, replace=False)
perm_q1 = permutation_importance(hgb_q1, X_q1_test.iloc[sample_idx], y_q1_test.iloc[sample_idx], 
                                 n_repeats=3, random_state=42, scoring='roc_auc')

hgb_feat_imp_q1 = pd.DataFrame({
    'Feature': X_q1.columns,
    'Importance': perm_q1.importances_mean
}).sort_values('Importance', ascending=False)
hgb_feat_imp_q1.to_csv('outputs/q1_feature_importance.csv', index=False)

print("\nTop 10 Risk Factors for Late Delivery (Odds Ratios):", flush=True)
print(odds_ratios_q1.head(10).to_string(index=False), flush=True)

print("\nTop 10 Feature Importances Q1 (Tree Permutation):", flush=True)
print(hgb_feat_imp_q1.head(10).to_string(index=False), flush=True)

# Top 5 highest risk shipping routes (min 100 orders)
df['route'] = df['seller_state'] + ' -> ' + df['customer_state']
route_risk = df.groupby('route').agg(
    order_count=('order_id', 'count'),
    late_rate=('is_late', 'mean'),
    median_delay_days=('delay_days', 'median'),
    median_transit_days=('carrier_transit_days', 'median'),
    low_review_rate=('is_low_review', 'mean')
).query('order_count >= 100').sort_values('late_rate', ascending=False).reset_index()

route_risk.to_csv('outputs/q1_top_risk_routes.csv', index=False)
print("\nTop 5 Highest Risk Shipping Routes:", flush=True)
print(route_risk.head(5).to_string(index=False), flush=True)

# Top 5 highest risk categories (min 100 orders)
cat_risk = df.groupby('primary_category').agg(
    order_count=('order_id', 'count'),
    late_rate=('is_late', 'mean'),
    median_weight_g=('total_weight_g', 'median'),
    median_volume_cm3=('total_volume_cm3', 'median'),
    low_review_rate=('is_low_review', 'mean')
).query('order_count >= 100').sort_values('late_rate', ascending=False).reset_index()

cat_risk.to_csv('outputs/q1_top_risk_categories.csv', index=False)
print("\nTop 5 Highest Risk Product Categories by Lateness:", flush=True)
print(cat_risk.head(5).to_string(index=False), flush=True)

# Bottleneck analysis: Decomposing fulfillment duration
bottlenecks = pd.DataFrame({
    'Metric': [
        'Approval Delay (Hours)',
        'Seller Handling / Dispatch Time (Days)',
        'Carrier Transit Time (Days)',
        'Total Delivery Days',
        'Estimated Buffer Days'
    ],
    'On-Time Orders (Median)': [
        df[df['is_late'] == 0]['approval_delay_hours'].median(),
        df[df['is_late'] == 0]['seller_handling_days'].median(),
        df[df['is_late'] == 0]['carrier_transit_days'].median(),
        df[df['is_late'] == 0]['actual_delivery_days'].median(),
        df[df['is_late'] == 0]['estimated_delivery_days'].median()
    ],
    'Late Orders (Median)': [
        df[df['is_late'] == 1]['approval_delay_hours'].median(),
        df[df['is_late'] == 1]['seller_handling_days'].median(),
        df[df['is_late'] == 1]['carrier_transit_days'].median(),
        df[df['is_late'] == 1]['actual_delivery_days'].median(),
        df[df['is_late'] == 1]['estimated_delivery_days'].median()
    ],
    'Difference (Late - OnTime)': [
        df[df['is_late'] == 1]['approval_delay_hours'].median() - df[df['is_late'] == 0]['approval_delay_hours'].median(),
        df[df['is_late'] == 1]['seller_handling_days'].median() - df[df['is_late'] == 0]['seller_handling_days'].median(),
        df[df['is_late'] == 1]['carrier_transit_days'].median() - df[df['is_late'] == 0]['carrier_transit_days'].median(),
        df[df['is_late'] == 1]['actual_delivery_days'].median() - df[df['is_late'] == 0]['actual_delivery_days'].median(),
        df[df['is_late'] == 1]['estimated_delivery_days'].median() - df[df['is_late'] == 0]['estimated_delivery_days'].median()
    ]
})
bottlenecks.to_csv('outputs/q1_bottleneck_breakdown.csv', index=False)
print("\nFulfillment Bottlenecks Decomposition (Median Days/Hours):", flush=True)
print(bottlenecks.to_string(index=False), flush=True)


# =========================================================================
# QUESTION 2: PREDICTING LOW REVIEW SCORE (review_score <= 2)
# =========================================================================
print("\n" + "="*60, flush=True)
print("QUESTION 2: PREDICTING LOW REVIEW SCORE (is_low_review)", flush=True)
print("="*60, flush=True)

df_q2 = df[df['is_low_review'].notnull()].copy()
df_q2['is_low_review'] = df_q2['is_low_review'].astype(int)

q2_features = [
    'delay_days', 'is_late', 'carrier_transit_days', 'actual_delivery_days',
    'seller_handling_days', 'is_seller_late_dispatch', 'approval_delay_hours',
    'estimated_delivery_days',
    'total_price', 'total_freight', 'freight_to_price_ratio',
    'total_weight_g', 'total_volume_cm3', 'item_count',
    'avg_photos_qty', 'avg_desc_length', 'max_installments',
    'distance_km', 'is_interstate', 'seller_orders', 'seller_avg_review'
]

q2_encoded = pd.get_dummies(df_q2[q2_features + ['primary_payment_type', 'customer_region', 'is_low_review']],
                           columns=['primary_payment_type', 'customer_region'], drop_first=True)

X_q2 = q2_encoded.drop(columns=['is_low_review'])
y_q2 = q2_encoded['is_low_review']

X_q2_train, X_q2_test, y_q2_train, y_q2_test = train_test_split(X_q2, y_q2, test_size=0.2, random_state=42, stratify=y_q2)

scaler_q2 = StandardScaler()
X_q2_train_sc = scaler_q2.fit_transform(X_q2_train)
X_q2_test_sc = scaler_q2.transform(X_q2_test)

log_reg_q2 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg_q2.fit(X_q2_train_sc, y_q2_train)
y_q2_prob_lr = log_reg_q2.predict_proba(X_q2_test_sc)[:, 1]

auc_lr_q2 = roc_auc_score(y_q2_test, y_q2_prob_lr)
ap_lr_q2 = average_precision_score(y_q2_test, y_q2_prob_lr)

odds_ratios_q2 = pd.DataFrame({
    'Feature': X_q2.columns,
    'Coefficient': log_reg_q2.coef_[0],
    'Odds_Ratio': np.exp(log_reg_q2.coef_[0])
}).sort_values('Odds_Ratio', ascending=False)
odds_ratios_q2.to_csv('outputs/q2_odds_ratios.csv', index=False)

print(f"Logistic Regression Q2 -> ROC-AUC: {auc_lr_q2:.4f}, PR-AUC: {ap_lr_q2:.4f}", flush=True)

hgb_q2 = HistGradientBoostingClassifier(random_state=42, class_weight='balanced', max_iter=100)
hgb_q2.fit(X_q2_train, y_q2_train)
y_q2_prob_hgb = hgb_q2.predict_proba(X_q2_test)[:, 1]
auc_hgb_q2 = roc_auc_score(y_q2_test, y_q2_prob_hgb)
ap_hgb_q2 = average_precision_score(y_q2_test, y_q2_prob_hgb)
print(f"HistGradientBoosting Q2 -> ROC-AUC: {auc_hgb_q2:.4f}, PR-AUC: {ap_hgb_q2:.4f}", flush=True)

sample_idx_q2 = np.random.RandomState(42).choice(len(X_q2_test), size=4000, replace=False)
perm_q2 = permutation_importance(hgb_q2, X_q2_test.iloc[sample_idx_q2], y_q2_test.iloc[sample_idx_q2], 
                                 n_repeats=3, random_state=42, scoring='roc_auc')
hgb_feat_imp_q2 = pd.DataFrame({
    'Feature': X_q2.columns,
    'Importance': perm_q2.importances_mean
}).sort_values('Importance', ascending=False)
hgb_feat_imp_q2.to_csv('outputs/q2_feature_importance.csv', index=False)

print("\nTop 10 Risk Factors for Low Review (Odds Ratios):", flush=True)
print(odds_ratios_q2.head(10).to_string(index=False), flush=True)

print("\nTop 10 Feature Importances Q2 (Tree Permutation):", flush=True)
print(hgb_feat_imp_q2.head(10).to_string(index=False), flush=True)

# =========================================================================
# QUESTION 3: ATTRIBUTABLE RISK & LATENESS CONTRIBUTION
# =========================================================================
print("\n" + "="*60, flush=True)
print("QUESTION 3: LATENESS ATTRIBUTABLE RISK & TEXT ANALYSIS", flush=True)
print("="*60, flush=True)

ontime_orders = df[df['is_late'] == 0]
late_orders = df[df['is_late'] == 1]

p_low_ontime = ontime_orders['is_low_review'].mean()
p_low_late = late_orders['is_low_review'].mean()
p_late = df['is_late'].mean()
p_low_overall = df['is_low_review'].mean()

rr = p_low_late / p_low_ontime
par = (p_low_overall - p_low_ontime) / p_low_overall

# McFadden's Pseudo-R2 for is_late alone in predicting is_low_review
from sklearn.linear_model import LogisticRegression
lr_null = LogisticRegression().fit(np.ones((len(df_q2), 1)), df_q2['is_low_review'])
null_log_loss = -roc_auc_score(df_q2['is_low_review'], lr_null.predict_proba(np.ones((len(df_q2), 1)))[:, 1]) # proxy

import statsmodels.api as sm
sm_null = sm.Logit(df_q2['is_low_review'], sm.add_constant(pd.Series(np.ones(len(df_q2)), index=df_q2.index))).fit(disp=False)
sm_late = sm.Logit(df_q2['is_low_review'], sm.add_constant(df_q2['is_late'])).fit(disp=False)
sm_delay = sm.Logit(df_q2['is_low_review'], sm.add_constant(df_q2[['is_late', 'delay_days']])).fit(disp=False)
sm_full = sm.Logit(df_q2['is_low_review'], sm.add_constant(X_q2.astype(float))).fit(disp=False)

mcfadden_late = 1.0 - (sm_late.llf / sm_null.llf)
mcfadden_delay = 1.0 - (sm_delay.llf / sm_null.llf)
mcfadden_full = 1.0 - (sm_full.llf / sm_null.llf)
pct_deviance_late = (sm_null.llf - sm_late.llf) / (sm_null.llf - sm_full.llf)

q3_metrics = pd.DataFrame({
    'Metric': [
        'Total Delivered Orders with Reviews',
        'Overall Low Review Rate (Score <= 2)',
        'On-Time Delivery Low Review Rate',
        'Late Delivery Low Review Rate',
        'Relative Risk (RR = P_late / P_ontime)',
        'Odds Ratio (Late vs On-time)',
        'Population Attributable Risk (PAR %)',
        'McFadden Pseudo-R2 (is_late alone)',
        'McFadden Pseudo-R2 (is_late + delay_days)',
        'McFadden Pseudo-R2 (Full Model)',
        '% of Model Deviance Explained by Delivery Lateness'
    ],
    'Value': [
        f"{len(df_q2):,}",
        f"{p_low_overall:.2%}",
        f"{p_low_ontime:.2%}",
        f"{p_low_late:.2%}",
        f"{rr:.2f}x",
        f"{(p_low_late/(1-p_low_late))/(p_low_ontime/(1-p_low_ontime)):.2f}",
        f"{par:.2%}",
        f"{mcfadden_late:.4f}",
        f"{mcfadden_delay:.4f}",
        f"{mcfadden_full:.4f}",
        f"{pct_deviance_late:.2%}"
    ]
})
q3_metrics.to_csv('outputs/q3_attributable_risk_metrics.csv', index=False)
print(q3_metrics.to_string(index=False), flush=True)

print("\nModel script completed successfully!", flush=True)
