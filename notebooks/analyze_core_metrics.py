import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()

# Query to extract master order analysis dataset
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
        AVG(review_score) as avg_review_score,
        COUNT(review_id) as review_count,
        MAX(CASE WHEN review_comment_message IS NOT NULL AND TRIM(review_comment_message) != '' THEN 1 ELSE 0 END) as has_review_comment,
        STRING_AGG(COALESCE(review_comment_message, ''), ' ') as all_comments
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
    
    r.has_review_comment,
    r.all_comments,

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

print("Running master query...")
df = con.execute(master_sql).df()
print(f"Dataset shape: {df.shape}")

# Calculate Haversine distance in km
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6367.0 * c
    return km

df['distance_km'] = haversine_np(df['customer_lng'], df['customer_lat'], df['seller_lng'], df['seller_lat'])
# Impute missing distance with median distance
df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())

# Calculate time intervals in days / hours
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['max_shipping_limit_date'] = pd.to_datetime(df['max_shipping_limit_date'])

# Time features
# 1. Total actual delivery days
df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
# 2. Estimated delivery days
df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
# 3. Delivery delay days (positive = late, negative = early)
df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 86400.0
# 4. Carrier transit days
df['carrier_transit_days'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.total_seconds() / 86400.0
# 5. Seller handling days
df['seller_handling_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / 86400.0
# 6. Seller late dispatch (carrier handoff after shipping limit date)
df['seller_dispatch_delay_days'] = (df['order_delivered_carrier_date'] - df['max_shipping_limit_date']).dt.total_seconds() / 86400.0
df['is_seller_late_dispatch'] = (df['seller_dispatch_delay_days'] > 0).astype(int)
# 7. Approval delay hours
df['approval_delay_hours'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600.0

# Temporal components
df['purchase_dow'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_year'] = df['order_purchase_timestamp'].dt.year

# Route
df['route'] = df['seller_state'] + ' -> ' + df['customer_state']
df['is_interstate'] = (df['seller_state'] != df['customer_state']).astype(int)

# Macro regions
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

print("\n--- Summary of Key Metrics ---")
print(f"Total delivered orders: {len(df):,}")
print(f"Late orders: {df['is_late'].sum():,} ({df['is_late'].mean():.2%})")
print(f"Low review orders (score<=2): {df['is_low_review'].sum():,} ({df['is_low_review'].mean():.2%})")
print(f"Median actual delivery days: {df['actual_delivery_days'].median():.1f} days")
print(f"Median estimated delivery days: {df['estimated_delivery_days'].median():.1f} days")
print(f"Median carrier transit days: {df['carrier_transit_days'].median():.1f} days")
print(f"Median seller handling days: {df['seller_handling_days'].median():.1f} days")
print(f"Seller late dispatch rate: {df['is_seller_late_dispatch'].mean():.2%}")

# Attributable risk calculation
late_rev = df.groupby('is_late')['is_low_review'].agg(['count', 'sum', 'mean']).reset_index()
late_rev['mean'] = late_rev['mean'] * 100.0
print("\n--- Low Reviews by Delivery Status ---")
print(late_rev)

p_low_ontime = df[df['is_late'] == 0]['is_low_review'].mean()
p_low_late = df[df['is_late'] == 1]['is_low_review'].mean()
p_late = df['is_late'].mean()
p_low_overall = df['is_low_review'].mean()
rr = p_low_late / p_low_ontime
par = (p_low_overall - p_low_ontime) / p_low_overall

print(f"\nP(Low Review | On-Time): {p_low_ontime:.2%}")
print(f"P(Low Review | Late): {p_low_late:.2%}")
print(f"Relative Risk (RR): {rr:.2f}x")
print(f"Population Attributable Risk (PAR / Attributable Fraction): {par:.2%}")

# Top 5 highest risk routes (min 100 orders)
route_stats = df.groupby('route').agg(
    order_count=('order_id', 'count'),
    late_rate=('is_late', 'mean'),
    avg_delay_days=('delay_days', 'mean'),
    median_transit_days=('carrier_transit_days', 'median'),
    low_review_rate=('is_low_review', 'mean')
).query('order_count >= 100').sort_values('late_rate', ascending=False)

print("\n--- Top 10 Highest Risk Shipping Routes (min 100 orders) ---")
print(route_stats.head(10))

# Top 5 highest risk categories (min 100 orders)
cat_stats = df.groupby('primary_category').agg(
    order_count=('order_id', 'count'),
    late_rate=('is_late', 'mean'),
    avg_weight=('total_weight_g', 'median'),
    avg_volume=('total_volume_cm3', 'median'),
    low_review_rate=('is_low_review', 'mean')
).query('order_count >= 100').sort_values('late_rate', ascending=False)

print("\n--- Top 10 Highest Risk Product Categories by Lateness (min 100 orders) ---")
print(cat_stats.head(10))
