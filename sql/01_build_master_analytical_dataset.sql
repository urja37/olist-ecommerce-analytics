-- =============================================================================
-- Script 01: Build Master Analytical Dataset for Delivery & Review Analysis
-- Engine: DuckDB / ANSI SQL
-- Description: Aggregates order items, payments, reviews, and geolocations
--              to construct an order-grain analytical table for predictive modeling.
-- =============================================================================

WITH geo_clean AS (
    -- Clean Brazil coordinates: bounding box lat [-34.0, 5.5], lng [-74.0, -34.0]
    SELECT 
        geolocation_zip_code_prefix AS zip_prefix,
        MEDIAN(geolocation_lat) AS lat,
        MEDIAN(geolocation_lng) AS lng
    FROM read_csv_auto('data/olist_geolocation_dataset.csv')
    WHERE geolocation_lat BETWEEN -34.0 AND 5.5
      AND geolocation_lng BETWEEN -74.0 AND -34.0
    GROUP BY geolocation_zip_code_prefix
),

order_items_agg AS (
    -- Aggregate order items to order grain
    SELECT 
        oi.order_id,
        COUNT(oi.order_item_id) AS item_count,
        COUNT(DISTINCT oi.product_id) AS unique_products,
        COUNT(DISTINCT oi.seller_id) AS unique_sellers,
        ANY_VALUE(oi.seller_id) AS primary_seller_id,
        SUM(oi.price) AS total_price,
        SUM(oi.freight_value) AS total_freight,
        MAX(oi.shipping_limit_date) AS max_shipping_limit_date,
        SUM(COALESCE(p.product_weight_g, 0)) AS total_weight_g,
        SUM(COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, 0)) AS total_volume_cm3,
        ANY_VALUE(COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')) AS primary_category,
        AVG(COALESCE(p.product_photos_qty, 1)) AS avg_photos_qty,
        AVG(COALESCE(p.product_description_lenght, 0)) AS avg_desc_length
    FROM read_csv_auto('data/olist_order_items_dataset.csv') oi
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p 
        ON oi.product_id = p.product_id
    LEFT JOIN read_csv_auto('data/product_category_name_translation.csv') t 
        ON p.product_category_name = t.product_category_name
    GROUP BY oi.order_id
),

payments_agg AS (
    -- Aggregate payment details
    SELECT 
        order_id,
        SUM(payment_value) AS total_payment_value,
        MAX(payment_installments) AS max_installments,
        ANY_VALUE(payment_type) AS primary_payment_type,
        COUNT(*) AS payment_installments_count
    FROM read_csv_auto('data/olist_order_payments_dataset.csv')
    GROUP BY order_id
),

reviews_agg AS (
    -- Aggregate review metrics
    SELECT 
        order_id,
        MIN(review_score) AS min_review_score,
        AVG(review_score) AS avg_review_score,
        COUNT(review_id) AS review_count,
        MAX(CASE WHEN review_comment_message IS NOT NULL AND TRIM(review_comment_message) != '' THEN 1 ELSE 0 END) AS has_review_comment,
        STRING_AGG(COALESCE(review_comment_message, ''), ' ') AS review_text
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
    
    -- Route and Interstate Indicator
    s.seller_state || ' -> ' || c.customer_state AS route,
    CASE WHEN s.seller_state != c.customer_state THEN 1 ELSE 0 END AS is_interstate,
    
    -- Timestamps
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    i.max_shipping_limit_date,
    
    -- Target 1: is_late (README Definition: delivered orders where actual delivery > estimated delivery)
    CASE 
        WHEN o.order_status = 'delivered' AND o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1
        WHEN o.order_status = 'delivered' THEN 0
        ELSE NULL
    END AS is_late,
    
    -- Target 2: is_low_review (README Definition: review_score <= 2)
    r.min_review_score AS review_score,
    CASE 
        WHEN r.min_review_score <= 2 THEN 1 
        WHEN r.min_review_score IS NOT NULL THEN 0 
        ELSE NULL 
    END AS is_low_review,
    
    -- Fulfillment Lifecycle Durations (Days / Hours)
    DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_delivered_customer_date) / (1000.0 * 86400.0) AS actual_delivery_days,
    DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_estimated_delivery_date) / (1000.0 * 86400.0) AS estimated_delivery_days,
    DATE_DIFF('millisecond', o.order_estimated_delivery_date, o.order_delivered_customer_date) / (1000.0 * 86400.0) AS delay_days,
    DATE_DIFF('millisecond', o.order_delivered_carrier_date, o.order_delivered_customer_date) / (1000.0 * 86400.0) AS carrier_transit_days,
    DATE_DIFF('millisecond', o.order_approved_at, o.order_delivered_carrier_date) / (1000.0 * 86400.0) AS seller_handling_days,
    DATE_DIFF('millisecond', i.max_shipping_limit_date, o.order_delivered_carrier_date) / (1000.0 * 86400.0) AS seller_dispatch_delay_days,
    CASE WHEN o.order_delivered_carrier_date > i.max_shipping_limit_date THEN 1 ELSE 0 END AS is_seller_late_dispatch,
    DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_approved_at) / (1000.0 * 3600.0) AS approval_delay_hours,

    -- Order & Product Complexity
    i.item_count,
    i.unique_products,
    i.unique_sellers,
    i.primary_seller_id,
    i.total_price,
    i.total_freight,
    i.total_freight / NULLIF(i.total_price, 0) AS freight_to_price_ratio,
    i.total_weight_g,
    i.total_volume_cm3,
    i.primary_category,
    i.avg_photos_qty,
    i.avg_desc_length,

    -- Payments
    p.total_payment_value,
    p.max_installments,
    p.primary_payment_type,

    -- Geolocation Coordinates
    c_geo.lat AS customer_lat,
    c_geo.lng AS customer_lng,
    s_geo.lat AS seller_lat,
    s_geo.lng AS seller_lng

FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_customers_dataset.csv') c 
    ON o.customer_id = c.customer_id
JOIN order_items_agg i 
    ON o.order_id = i.order_id
LEFT JOIN read_csv_auto('data/olist_sellers_dataset.csv') s 
    ON i.primary_seller_id = s.seller_id
LEFT JOIN payments_agg p 
    ON o.order_id = p.order_id
LEFT JOIN reviews_agg r 
    ON o.order_id = r.order_id
LEFT JOIN geo_clean c_geo 
    ON c.customer_zip_code_prefix = c_geo.zip_prefix
LEFT JOIN geo_clean s_geo 
    ON s.seller_zip_code_prefix = s_geo.zip_prefix
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL;
