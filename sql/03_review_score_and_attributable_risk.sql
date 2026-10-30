-- =============================================================================
-- Script 03: Review Score Attributable Risk & Lateness Contribution
-- Engine: DuckDB / ANSI SQL
-- Description: Quantifies low review rates (review_score <= 2) across on-time
--              vs. late deliveries, computing Relative Risk and Population Attributable Risk.
-- =============================================================================

WITH order_level AS (
    SELECT 
        o.order_id,
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
            ELSE 0 
        END AS is_late,
        r.review_score,
        CASE 
            WHEN r.review_score <= 2 THEN 1 
            ELSE 0 
        END AS is_low_review
    FROM read_csv_auto('data/olist_orders_dataset.csv') o
    JOIN read_csv_auto('data/olist_order_reviews_dataset.csv') r 
        ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND r.review_score IS NOT NULL
),
summary_stats AS (
    SELECT 
        is_late,
        COUNT(*) AS total_orders,
        SUM(is_low_review) AS low_review_orders,
        AVG(is_low_review) AS low_review_rate
    FROM order_level
    GROUP BY is_late
)
SELECT 
    ontime.total_orders AS ontime_orders,
    ontime.low_review_orders AS ontime_low_reviews,
    ROUND(ontime.low_review_rate * 100, 2) AS ontime_low_review_rate_pct,
    
    late.total_orders AS late_orders,
    late.low_review_orders AS late_low_reviews,
    ROUND(late.low_review_rate * 100, 2) AS late_low_review_rate_pct,
    
    -- Relative Risk: P(Low Review | Late) / P(Low Review | On-Time)
    ROUND(late.low_review_rate / ontime.low_review_rate, 2) AS relative_risk,
    
    -- Odds Ratio: [P_late / (1 - P_late)] / [P_ontime / (1 - P_ontime)]
    ROUND((late.low_review_rate / (1.0 - late.low_review_rate)) / 
          (ontime.low_review_rate / (1.0 - ontime.low_review_rate)), 2) AS odds_ratio,
    
    -- Population Attributable Risk (PAR %): (Total_Low_Rate - OnTime_Low_Rate) / Total_Low_Rate
    ROUND((( (ontime.low_review_orders + late.low_review_orders) * 1.0 / (ontime.total_orders + late.total_orders) - ontime.low_review_rate ) /
           ( (ontime.low_review_orders + late.low_review_orders) * 1.0 / (ontime.total_orders + late.total_orders) )) * 100, 2) AS population_attributable_risk_pct

FROM summary_stats ontime
CROSS JOIN summary_stats late
WHERE ontime.is_late = 0 
  AND late.is_late = 1;
