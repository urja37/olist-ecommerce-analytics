-- =============================================================================
-- Script 02: Route & Product Category Lateness Risk Analysis
-- Engine: DuckDB / ANSI SQL
-- Description: Aggregates shipping routes and product categories to identify
--              the top failure points and operational corridors driving delivery delays.
-- =============================================================================

-- 1. High-Risk Shipping Routes (Min 100 Orders)
SELECT 
    s.seller_state || ' -> ' || c.customer_state AS route,
    s.seller_state,
    c.customer_state,
    COUNT(o.order_id) AS total_orders,
    SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_orders,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END) * 100, 2) AS lateness_rate_pct,
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_delivered_carrier_date, o.order_delivered_customer_date) / (1000.0 * 86400.0)), 1) AS median_carrier_transit_days,
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_estimated_delivery_date) / (1000.0 * 86400.0)), 1) AS median_promised_buffer_days,
    ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0.0 END) * 100, 2) AS low_review_rate_pct
FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_customers_dataset.csv') c ON o.customer_id = c.customer_id
JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi ON o.order_id = oi.order_id
JOIN read_csv_auto('data/olist_sellers_dataset.csv') s ON oi.seller_id = s.seller_id
LEFT JOIN read_csv_auto('data/olist_order_reviews_dataset.csv') r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY s.seller_state, c.customer_state
HAVING COUNT(o.order_id) >= 100
ORDER BY lateness_rate_pct DESC
LIMIT 15;

-- 2. High-Risk Product Categories by Lateness (Min 100 Orders)
SELECT 
    COALESCE(t.product_category_name_english, p.product_category_name, 'Unknown') AS category,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END) * 100, 2) AS lateness_rate_pct,
    ROUND(MEDIAN(p.product_weight_g), 0) AS median_weight_g,
    ROUND(MEDIAN(p.product_length_cm * p.product_height_cm * p.product_width_cm), 0) AS median_volume_cm3,
    ROUND(AVG(oi.freight_value), 2) AS avg_freight_value,
    ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0.0 END) * 100, 2) AS low_review_rate_pct
FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi ON o.order_id = oi.order_id
JOIN read_csv_auto('data/olist_products_dataset.csv') p ON oi.product_id = p.product_id
LEFT JOIN read_csv_auto('data/product_category_name_translation.csv') t ON p.product_category_name = t.product_category_name
LEFT JOIN read_csv_auto('data/olist_order_reviews_dataset.csv') r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY 1
HAVING COUNT(DISTINCT o.order_id) >= 100
ORDER BY lateness_rate_pct DESC
LIMIT 15;
