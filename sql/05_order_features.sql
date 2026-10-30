-- =============================================================================
-- Script: sql/05_order_features.sql
-- Database: olist.duckdb
-- Stage: Master Order Features with Calculated Columns
-- Purpose: Join delivered orders with customers, items, sellers, geolocation 
--          centroids, reviews, and categories, deriving core operational features:
--          delivery delays, actual vs estimated durations, seller handling,
--          Haversine distance, freight ratio, regional flags, and review labels.
-- =============================================================================

CREATE OR REPLACE TABLE order_features AS
SELECT 
    -- Order identifiers and timestamps
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    
    -- Calculated delivery performance features
    CASE 
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
        ELSE 0 
    END AS is_late,
    
    ROUND(DATE_DIFF('second', o.order_estimated_delivery_date, o.order_delivered_customer_date) / 86400.0, 2) AS days_late,
    ROUND(DATE_DIFF('second', o.order_purchase_timestamp, o.order_delivered_customer_date) / 86400.0, 2) AS actual_days,
    ROUND(DATE_DIFF('second', o.order_purchase_timestamp, o.order_estimated_delivery_date) / 86400.0, 2) AS estimated_days,
    ROUND(DATE_DIFF('second', o.order_purchase_timestamp, o.order_delivered_carrier_date) / 86400.0, 2) AS seller_handling_days,
    
    -- Customer attributes
    cust.customer_zip_code_prefix AS customer_zip,
    cust.customer_city,
    cust.customer_state,
    
    -- Aggregated order items attributes (from items_agg)
    i.n_items,
    i.n_sellers,
    i.total_price,
    i.total_freight,
    ROUND(i.total_freight / NULLIF(i.total_price, 0), 4) AS freight_ratio,
    i.total_weight_g,
    i.total_volume_cm3,
    i.main_seller_id,
    i.category_pt,
    COALESCE(tr.product_category_name_english, i.category_pt) AS category_en,
    
    -- Main seller attributes
    sell.seller_zip_code_prefix AS seller_zip,
    sell.seller_city,
    sell.seller_state,
    
    -- Interstate vs Intrastate route
    CASE 
        WHEN cust.customer_state = sell.seller_state THEN 1 
        ELSE 0 
    END AS same_state,
    
    -- Geolocation coordinates (from geo_zip)
    c.lat AS customer_lat,
    c.lng AS customer_lng,
    s.lat AS seller_lat,
    s.lng AS seller_lng,
    
    -- Haversine shipping distance formula (km)
    ROUND(2 * 6371 * ASIN(SQRT(
      POWER(SIN(RADIANS(c.lat - s.lat)/2), 2) +
      COS(RADIANS(s.lat)) * COS(RADIANS(c.lat)) *
      POWER(SIN(RADIANS(c.lng - s.lng)/2), 2)
    )), 2) AS distance_km,
    
    -- Temporal features from purchase timestamp
    EXTRACT(MONTH FROM o.order_purchase_timestamp) AS purchase_month,
    EXTRACT(DOW FROM o.order_purchase_timestamp) AS purchase_dow,
    
    -- Customer review score and low review target
    r.review_score,
    CASE 
        WHEN r.review_score <= 2 THEN 1 
        WHEN r.review_score IS NOT NULL THEN 0 
        ELSE NULL 
    END AS low_review

FROM orders o
JOIN customers cust 
    ON o.customer_id = cust.customer_id
LEFT JOIN items_agg i 
    ON o.order_id = i.order_id
LEFT JOIN sellers sell 
    ON i.main_seller_id = sell.seller_id
LEFT JOIN geo_zip c 
    ON cust.customer_zip_code_prefix = c.zip
LEFT JOIN geo_zip s 
    ON sell.seller_zip_code_prefix = s.zip
LEFT JOIN reviews_clean r 
    ON o.order_id = r.order_id
LEFT JOIN category_tr tr 
    ON i.category_pt = tr.product_category_name
WHERE o.order_status = 'delivered';

-- -----------------------------------------------------------------------------
-- Verification & Data Integrity Diagnostics
-- -----------------------------------------------------------------------------

-- 1. Table structure, grain, and record counts
-- Grain check: Verified one row per order (COUNT(*) = COUNT(DISTINCT order_id) = 96,478; 0 duplicate orders)
SELECT 
    COUNT(*) AS total_delivered_orders,
    COUNT(DISTINCT order_id) AS unique_orders,
    COUNT(distance_km) AS matched_distance,
    COUNT(review_score) AS matched_reviews,
    ROUND(AVG(is_late) * 100, 2) AS lateness_pct,
    ROUND(AVG(low_review) * 100, 2) AS low_review_pct,
    ROUND(AVG(same_state) * 100, 2) AS same_state_pct
FROM order_features;

-- 2. Inspect sample records of calculated columns
SELECT 
    order_id,
    customer_state,
    seller_state,
    same_state,
    distance_km,
    actual_days,
    estimated_days,
    days_late,
    is_late,
    seller_handling_days,
    freight_ratio,
    purchase_month,
    purchase_dow,
    review_score,
    low_review
FROM order_features
LIMIT 5;

-- 3. Summary statistics of continuous calculated features
SELECT 
    'distance_km' AS feature,
    ROUND(MIN(distance_km), 2) AS min_val,
    ROUND(MEDIAN(distance_km), 2) AS median_val,
    ROUND(AVG(distance_km), 2) AS avg_val,
    ROUND(MAX(distance_km), 2) AS max_val
FROM order_features
UNION ALL
SELECT 
    'actual_days',
    ROUND(MIN(actual_days), 2),
    ROUND(MEDIAN(actual_days), 2),
    ROUND(AVG(actual_days), 2),
    ROUND(MAX(actual_days), 2)
FROM order_features
UNION ALL
SELECT 
    'estimated_days',
    ROUND(MIN(estimated_days), 2),
    ROUND(MEDIAN(estimated_days), 2),
    ROUND(AVG(estimated_days), 2),
    ROUND(MAX(estimated_days), 2)
FROM order_features
UNION ALL
SELECT 
    'days_late',
    ROUND(MIN(days_late), 2),
    ROUND(MEDIAN(days_late), 2),
    ROUND(AVG(days_late), 2),
    ROUND(MAX(days_late), 2)
FROM order_features
UNION ALL
SELECT 
    'seller_handling_days',
    ROUND(MIN(seller_handling_days), 2),
    ROUND(MEDIAN(seller_handling_days), 2),
    ROUND(AVG(seller_handling_days), 2),
    ROUND(MAX(seller_handling_days), 2)
FROM order_features
UNION ALL
SELECT 
    'freight_ratio',
    ROUND(MIN(freight_ratio), 4),
    ROUND(MEDIAN(freight_ratio), 4),
    ROUND(AVG(freight_ratio), 4),
    ROUND(MAX(freight_ratio), 4)
FROM order_features;
