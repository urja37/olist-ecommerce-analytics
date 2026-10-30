-- =============================================================================
-- Script: sql/01_data_checks.sql
-- Database: olist.duckdb
-- Purpose: Initial data sanity checks and data hygiene audit
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Check 1: Row count per table
-- -----------------------------------------------------------------------------
SELECT 'orders' AS table_name, COUNT(*) AS total_rows FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'reviews', COUNT(*) FROM reviews
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'geo', COUNT(*) FROM geo
UNION ALL
SELECT 'payments', COUNT(*) FROM payments
UNION ALL
SELECT 'category_tr', COUNT(*) FROM category_tr;

-- -----------------------------------------------------------------------------
-- Check 2: Orders per order_status
-- -----------------------------------------------------------------------------
SELECT 
    order_status,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_share
FROM orders
GROUP BY order_status
ORDER BY order_count DESC;

-- -----------------------------------------------------------------------------
-- Check 3: Date range of orders (purchase, delivery, and estimated dates)
-- -----------------------------------------------------------------------------
SELECT 
    MIN(order_purchase_timestamp) AS earliest_purchase,
    MAX(order_purchase_timestamp) AS latest_purchase,
    MIN(order_delivered_customer_date) AS earliest_delivery,
    MAX(order_delivered_customer_date) AS latest_delivery,
    MIN(order_estimated_delivery_date) AS earliest_estimated,
    MAX(order_estimated_delivery_date) AS latest_estimated
FROM orders;

-- -----------------------------------------------------------------------------
-- Check 4: Orders with more than one review
-- -----------------------------------------------------------------------------
SELECT 
    order_id,
    COUNT(*) AS review_count,
    MIN(review_score) AS min_review_score,
    MAX(review_score) AS max_review_score
FROM reviews
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY review_count DESC, order_id;

-- Summary count of multi-review orders:
SELECT 
    review_count,
    COUNT(*) AS number_of_orders
FROM (
    SELECT order_id, COUNT(*) AS review_count
    FROM reviews
    GROUP BY order_id
    HAVING COUNT(*) > 1
) sub
GROUP BY review_count
ORDER BY review_count DESC;

-- -----------------------------------------------------------------------------
-- Check 5: Delivered orders with missing delivery date
-- -----------------------------------------------------------------------------
SELECT 
    order_id,
    order_status,
    order_purchase_timestamp,
    order_estimated_delivery_date,
    order_delivered_customer_date
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NULL;

-- -----------------------------------------------------------------------------
-- Check 6: Logical timestamp anomalies (delivery earlier than purchase)
-- -----------------------------------------------------------------------------
SELECT 
    order_id,
    order_status,
    order_purchase_timestamp,
    order_delivered_customer_date,
    DATE_DIFF('second', order_purchase_timestamp, order_delivered_customer_date) AS diff_seconds
FROM orders
WHERE order_delivered_customer_date < order_purchase_timestamp;

-- Additional integrity check: Carrier handoff earlier than purchase timestamp
SELECT 
    order_id,
    order_status,
    order_purchase_timestamp,
    order_delivered_carrier_date
FROM orders
WHERE order_delivered_carrier_date < order_purchase_timestamp;
