-- =============================================================================
-- Script: sql/07_order_master_clean.sql
-- Database: olist.duckdb
-- Stage: Step 18 - Clean Master Analytical Dataset
-- Purpose: Apply business integrity filters, remove impossible timestamp values
--          and zero weights, cap extreme 99th percentile outliers, and create
--          the production-ready 'order_master' table.
-- =============================================================================

CREATE OR REPLACE TABLE order_master AS
WITH p99_thresholds AS (
    SELECT 
        QUANTILE_CONT(total_weight_g, 0.99) AS p99_weight,
        QUANTILE_CONT(freight_ratio, 0.99) AS p99_freight_ratio,
        QUANTILE_CONT(total_volume_cm3, 0.99) AS p99_volume
    FROM order_features
    WHERE total_weight_g > 0
      AND distance_km IS NOT NULL
      AND order_delivered_customer_date IS NOT NULL
      AND order_delivered_carrier_date >= order_purchase_timestamp
      AND order_delivered_customer_date >= order_delivered_carrier_date
)
SELECT 
    f.*,
    
    -- Capped extreme outliers (Winsorized at 99th percentile)
    LEAST(f.total_weight_g, t.p99_weight) AS weight_capped,
    LEAST(f.freight_ratio, t.p99_freight_ratio) AS freight_ratio_capped,
    LEAST(f.total_volume_cm3, t.p99_volume) AS volume_capped,
    
    -- Log-transformed features for skewed distributions
    LN(1 + f.total_weight_g) AS log_weight,
    LN(1 + f.total_volume_cm3) AS log_volume,
    LN(1 + f.total_price) AS log_price,
    LN(1 + f.total_freight) AS log_freight,
    LN(1 + f.distance_km) AS log_distance

FROM order_features f
CROSS JOIN p99_thresholds t
WHERE 
    -- 1. Exclude missing key fields
    f.order_delivered_customer_date IS NOT NULL
    AND f.order_delivered_carrier_date IS NOT NULL
    AND f.order_approved_at IS NOT NULL
    AND f.distance_km IS NOT NULL
    AND f.total_weight_g IS NOT NULL
    
    -- 2. Exclude impossible values
    AND f.total_weight_g > 0
    AND f.actual_days >= 0
    AND f.seller_handling_days >= 0
    AND (f.order_delivered_customer_date >= f.order_delivered_carrier_date);

-- -----------------------------------------------------------------------------
-- Verification & Data Hygiene Diagnostics
-- -----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS clean_order_count,
    COUNT(DISTINCT order_id) AS unique_orders,
    ROUND(AVG(is_late) * 100, 2) AS clean_late_rate_pct,
    ROUND(AVG(low_review) * 100, 2) AS clean_low_review_pct,
    ROUND(MAX(weight_capped), 1) AS max_weight_capped_g,
    ROUND(MAX(freight_ratio_capped), 4) AS max_freight_ratio_capped
FROM order_master;
