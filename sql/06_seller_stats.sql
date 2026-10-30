-- =============================================================================
-- Script: sql/06_seller_stats.sql
-- Database: olist.duckdb
-- Stage: Seller-Level Analytics & Route Risk Profiling
-- Purpose: Aggregate seller performance metrics, demonstrate window functions,
--          rank merchants by late delivery rate, and profile state & route corridors.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Seller Performance View (seller_stats)
-- Aggregates orders, late rate, review score, and handling days per seller.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW seller_stats AS
SELECT 
    main_seller_id AS seller_id,
    seller_state,
    seller_city,
    COUNT(order_id) AS n_orders,
    SUM(is_late) AS n_late_orders,
    ROUND(AVG(is_late) * 100.0, 2) AS late_rate_pct,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    ROUND(AVG(seller_handling_days), 2) AS avg_handling_days,
    ROUND(AVG(days_late), 2) AS avg_days_late,
    ROUND(SUM(total_price), 2) AS total_sales_volume
FROM order_features
GROUP BY main_seller_id, seller_state, seller_city;

-- -----------------------------------------------------------------------------
-- Query 1: Rank Sellers by Late Rate (Minimum 30 Orders)
-- Window functions demonstrate:
--   - RANK() and DENSE_RANK() for national performance ranking
--   - PERCENT_RANK() for relative percentile placement
--   - NTILE(4) for operational risk tiering (Quartile 1 = highest late risk)
--   - PARTITION BY window average to benchmark vs. state peers
-- -----------------------------------------------------------------------------
WITH qualified_sellers AS (
    SELECT * 
    FROM seller_stats
    WHERE n_orders >= 30
)
SELECT 
    -- Window functions
    RANK() OVER (ORDER BY late_rate_pct DESC, n_orders DESC) AS seller_late_rank,
    DENSE_RANK() OVER (ORDER BY late_rate_pct DESC) AS seller_late_dense_rank,
    
    seller_id,
    seller_state,
    seller_city,
    n_orders,
    n_late_orders,
    late_rate_pct,
    avg_review_score,
    avg_handling_days,
    
    -- State benchmark window function
    ROUND(AVG(late_rate_pct) OVER (PARTITION BY seller_state), 2) AS state_peer_avg_late_rate,
    ROUND(late_rate_pct - AVG(late_rate_pct) OVER (PARTITION BY seller_state), 2) AS diff_from_state_avg,
    
    -- Percentile and quartile
    ROUND(PERCENT_RANK() OVER (ORDER BY late_rate_pct), 4) AS late_percentile,
    NTILE(4) OVER (ORDER BY late_rate_pct DESC) AS risk_quartile

FROM qualified_sellers
ORDER BY late_rate_pct DESC, n_orders DESC
LIMIT 15;

-- -----------------------------------------------------------------------------
-- Query 2: Late Delivery Rate by Seller Origin State
-- Aggregates order volume and calculates state-level ranking and cumulative share.
-- -----------------------------------------------------------------------------
SELECT 
    seller_state,
    COUNT(order_id) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(AVG(is_late) * 100.0, 2) AS late_rate_pct,
    ROUND(AVG(seller_handling_days), 2) AS avg_handling_days,
    ROUND(AVG(days_late), 2) AS avg_days_late,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    -- Window functions: State rank and cumulative volume share
    RANK() OVER (ORDER BY AVG(is_late) DESC) AS state_late_rank,
    ROUND(SUM(COUNT(order_id)) OVER (ORDER BY COUNT(order_id) DESC) * 100.0 / 
          SUM(COUNT(order_id)) OVER (), 2) AS cumulative_volume_pct
FROM order_features
GROUP BY seller_state
HAVING COUNT(order_id) >= 30
ORDER BY late_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 3: Late Delivery Rate by Route (Seller State -> Customer State)
-- Identifies top logistical bottleneck corridors across Brazil (Min 30 orders).
-- -----------------------------------------------------------------------------
SELECT 
    seller_state || ' -> ' || customer_state AS route,
    seller_state,
    customer_state,
    same_state,
    COUNT(order_id) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(AVG(is_late) * 100.0, 2) AS late_rate_pct,
    ROUND(AVG(distance_km), 1) AS avg_distance_km,
    ROUND(AVG(days_late), 2) AS avg_days_late,
    ROUND(AVG(seller_handling_days), 2) AS avg_handling_days,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    -- Window functions: Route rank
    RANK() OVER (ORDER BY AVG(is_late) DESC) AS route_late_rank,
    DENSE_RANK() OVER (ORDER BY AVG(is_late) DESC) AS route_late_dense_rank
FROM order_features
GROUP BY seller_state, customer_state, same_state
HAVING COUNT(order_id) >= 30
ORDER BY late_rate_pct DESC
LIMIT 15;
