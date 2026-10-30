-- =============================================================================
-- Script: sql/03_reviews.sql
-- Database: olist.duckdb
-- Stage: Collapse Reviews to One Per Order
-- Purpose: Deduplicate customer reviews by selecting the most recent review
--          per order using DuckDB's QUALIFY clause.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Step Implementation: Create reviews_clean table
-- Resolves multi-review orders (547 orders with 2-3 reviews) by ordering by
-- review_creation_date DESC to keep the most recent review score.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE reviews_clean AS
SELECT order_id, review_score
FROM reviews
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY order_id ORDER BY review_creation_date DESC) = 1;

-- -----------------------------------------------------------------------------
-- Verification & Quality Assurance Diagnostics
-- -----------------------------------------------------------------------------

-- 1. Deduplication summary: compare raw reviews vs. reviews_clean
SELECT 
    'Raw Reviews' AS stage,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT order_id) AS unique_orders,
    COUNT(*) - COUNT(DISTINCT order_id) AS duplicate_rows
FROM reviews
UNION ALL
SELECT 
    'Cleaned Reviews (reviews_clean)' AS stage,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT order_id) AS unique_orders,
    COUNT(*) - COUNT(DISTINCT order_id) AS duplicate_rows
FROM reviews_clean;

-- 2. Verify score distribution and low review share in reviews_clean
SELECT 
    review_score,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_share,
    CASE WHEN review_score <= 2 THEN 'Low Review (1-2)' ELSE 'Satisfied (3-5)' END AS review_tier
FROM reviews_clean
GROUP BY review_score
ORDER BY review_score;

-- 3. Confirm 1-to-1 cardinality with orders table
SELECT 
    COUNT(o.order_id) AS total_orders,
    COUNT(rc.order_id) AS orders_with_review,
    COUNT(o.order_id) - COUNT(rc.order_id) AS orders_without_review,
    ROUND(COUNT(rc.order_id) * 100.0 / COUNT(o.order_id), 2) AS review_coverage_pct
FROM orders o
LEFT JOIN reviews_clean rc ON o.order_id = rc.order_id;
