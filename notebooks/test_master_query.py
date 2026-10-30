import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()

query = """
WITH geo_clean AS (
    -- Clean geolocation: filter bounding box for Brazil lat: [-34, 5], lng: [-74, -34]
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
        MIN(oi.shipping_limit_date) as min_shipping_limit_date,
        SUM(COALESCE(p.product_weight_g, 0)) as total_weight_g,
        SUM(COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, 0)) as total_volume_cm3,
        ANY_VALUE(COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')) as primary_category,
        AVG(COALESCE(p.product_photos_qty, 1)) as avg_photos_qty
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
        ANY_VALUE(payment_type) as primary_payment_type,
        COUNT(*) as payment_count
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
        ANY_VALUE(review_comment_message) as sample_comment
    FROM read_csv_auto('data/olist_order_reviews_dataset.csv')
    GROUP BY order_id
)
SELECT 
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    
    -- Targets
    CASE 
        WHEN o.order_status = 'delivered' AND o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1
        WHEN o.order_status = 'delivered' THEN 0
        ELSE NULL
    END as is_late,
    
    r.min_review_score as review_score,
    CASE 
        WHEN r.min_review_score <= 2 THEN 1 
        WHEN r.min_review_score IS NOT NULL THEN 0 
        ELSE NULL 
    END as is_low_review,
    
    r.has_review_comment,
    r.sample_comment,

    -- Items & Product features
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
    i.max_shipping_limit_date,

    -- Payments features
    p.total_payment_value,
    p.max_installments,
    p.primary_payment_type,

    -- Geolocation coordinates
    c_geo.lat as customer_lat,
    c_geo.lng as customer_lng,
    s_geo.lat as seller_lat,
    s_geo.lng as seller_lng

FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_customers_dataset.csv') c ON o.customer_id = c.customer_id
LEFT JOIN order_items_agg i ON o.order_id = i.order_id
LEFT JOIN read_csv_auto('data/olist_sellers_dataset.csv') s ON i.primary_seller_id = s.seller_id
LEFT JOIN payments_agg p ON o.order_id = p.order_id
LEFT JOIN reviews_agg r ON o.order_id = r.order_id
LEFT JOIN geo_clean c_geo ON c.customer_zip_code_prefix = c_geo.zip_prefix
LEFT JOIN geo_clean s_geo ON s.seller_zip_code_prefix = s_geo.zip_prefix
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
"""

df_test = con.execute(query).df()
print(f"Delivered orders extracted: {len(df_test):,}")
print("Null values per column:")
print(df_test.isnull().sum()[df_test.isnull().sum() > 0])
print("\nis_late counts:")
print(df_test['is_late'].value_counts(normalize=True))
print("\nis_low_review counts:")
print(df_test['is_low_review'].value_counts(dropna=False, normalize=True))
