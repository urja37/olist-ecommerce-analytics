-- =============================================================================
-- Script 04: Fulfillment Bottlenecks Lifecycle Decomposition
-- Engine: DuckDB / ANSI SQL
-- Description: Decomposes the end-to-end order journey into approval delay,
--              seller handling time, and carrier transit duration.
-- =============================================================================

SELECT 
    CASE 
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'Late Deliveries'
        ELSE 'On-Time Deliveries'
    END AS delivery_cohort,
    COUNT(*) AS order_count,
    
    -- Approval delay (hours)
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_approved_at) / (1000.0 * 3600.0)), 2) AS median_approval_hours,
    
    -- Seller dispatch handling time (days)
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_approved_at, o.order_delivered_carrier_date) / (1000.0 * 86400.0)), 2) AS median_seller_handling_days,
    
    -- Seller dispatch SLA compliance
    ROUND(AVG(CASE WHEN o.order_delivered_carrier_date > oi_max.max_shipping_limit_date THEN 1.0 ELSE 0.0 END) * 100, 2) AS seller_late_dispatch_pct,
    
    -- Carrier transit duration (days)
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_delivered_carrier_date, o.order_delivered_customer_date) / (1000.0 * 86400.0)), 2) AS median_carrier_transit_days,
    
    -- Total actual delivery duration (days)
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_delivered_customer_date) / (1000.0 * 86400.0)), 2) AS median_actual_delivery_days,
    
    -- Promised delivery buffer (days)
    ROUND(MEDIAN(DATE_DIFF('millisecond', o.order_purchase_timestamp, o.order_estimated_delivery_date) / (1000.0 * 86400.0)), 2) AS median_promised_buffer_days

FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN (
    SELECT order_id, MAX(shipping_limit_date) AS max_shipping_limit_date
    FROM read_csv_auto('data/olist_order_items_dataset.csv')
    GROUP BY order_id
) oi_max ON o.order_id = oi_max.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY 1
ORDER BY delivery_cohort DESC;
