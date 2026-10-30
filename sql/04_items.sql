-- =============================================================================
-- Script: sql/04_items.sql
-- Database: olist.duckdb
-- Stage: Collapse Items to One Row Per Order
-- Purpose: Pre-aggregate order_items and products to order grain before joining
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Step Implementation: Create items_agg table
-- Pre-aggregates item count, seller count, price, freight, weight, and volume.
-- Simplification note: MIN() is applied to seller_id and category_name as a 
-- deterministic, lightweight method to select a representative main seller and 
-- category for multi-item orders.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE items_agg AS
SELECT
  i.order_id,
  COUNT(*)                         AS n_items,
  COUNT(DISTINCT i.seller_id)      AS n_sellers,
  SUM(i.price)                     AS total_price,
  SUM(i.freight_value)             AS total_freight,
  SUM(p.product_weight_g)          AS total_weight_g,
  SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS total_volume_cm3,
  MIN(i.seller_id)                 AS main_seller_id,
  MIN(p.product_category_name)     AS category_pt
FROM order_items i
JOIN products p USING (product_id)
GROUP BY i.order_id;

-- -----------------------------------------------------------------------------
-- Verification & Quality Assurance Diagnostics
-- -----------------------------------------------------------------------------

-- 1. Aggregation summary & cardinality verification
SELECT 
    COUNT(*) AS total_orders,
    COUNT(DISTINCT order_id) AS unique_orders,
    ROUND(AVG(n_items), 2) AS avg_items_per_order,
    MAX(n_items) AS max_items_per_order,
    ROUND(AVG(n_sellers), 2) AS avg_sellers_per_order,
    MAX(n_sellers) AS max_sellers_per_order,
    ROUND(AVG(total_price), 2) AS avg_order_price,
    ROUND(AVG(total_freight), 2) AS avg_order_freight
FROM items_agg;

-- 2. Multi-item and multi-seller distribution
SELECT 
    CASE 
        WHEN n_items = 1 THEN 'Single-item order'
        ELSE 'Multi-item order (>=2 items)'
    END AS item_type,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_share
FROM items_agg
GROUP BY 1;

SELECT 
    CASE 
        WHEN n_sellers = 1 THEN 'Single seller'
        ELSE 'Multiple sellers (>=2 sellers)'
    END AS seller_type,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_share
FROM items_agg
GROUP BY 1;

-- 3. Match coverage with orders table
SELECT 
    COUNT(o.order_id) AS total_orders,
    COUNT(ia.order_id) AS orders_with_items,
    COUNT(o.order_id) - COUNT(ia.order_id) AS orders_without_items
FROM orders o
LEFT JOIN items_agg ia ON o.order_id = ia.order_id;
