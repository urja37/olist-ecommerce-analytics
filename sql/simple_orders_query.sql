-- ============================================================================
-- SQL: Simple Order Analysis Dataset
-- Purpose: Extract delivered orders with key fulfillment milestones & review scores
-- Level: Entry-Level Data Analyst / Junior Data Scientist
-- ============================================================================

WITH items_agg AS (
    -- Aggregate item level info up to order level
    SELECT 
        order_id,
        COUNT(order_item_id) AS total_items,
        SUM(price) AS total_price,
        SUM(freight_value) AS total_freight,
        ANY_VALUE(seller_id) AS main_seller_id,
        SUM(COALESCE(product_weight_g, 0)) AS total_weight_g
    FROM read_csv_auto('data/olist_order_items_dataset.csv') oi
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p 
        ON oi.product_id = p.product_id
    GROUP BY order_id
),
reviews_clean AS (
    -- Deduplicate reviews: take the latest review score per order
    SELECT 
        order_id,
        review_score,
        review_creation_date
    FROM (
        SELECT 
            order_id,
            review_score,
            review_creation_date,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_creation_date DESC) AS rn
        FROM read_csv_auto('data/olist_order_reviews_dataset.csv')
    )
    WHERE rn = 1
)
SELECT 
    o.order_id,
    o.customer_id,
    c.customer_state,
    
    -- Timestamps
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    
    -- Fulfillment Milestones (in days)
    DATEDIFF('day', o.order_purchase_timestamp, o.order_delivered_customer_date) AS actual_delivery_days,
    DATEDIFF('day', o.order_purchase_timestamp, o.order_estimated_delivery_date) AS promised_delivery_days,
    DATEDIFF('day', o.order_approved_at, o.order_delivered_carrier_date) AS seller_handling_days,
    DATEDIFF('day', o.order_delivered_carrier_date, o.order_delivered_customer_date) AS carrier_transit_days,
    
    -- Target Metric: Is the order late? (1 = Late, 0 = On-Time)
    CASE 
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
        ELSE 0 
    END AS is_late,
    
    -- Customer Satisfaction
    r.review_score,
    CASE 
        WHEN r.review_score <= 2 THEN 1 
        WHEN r.review_score IS NOT NULL THEN 0 
        ELSE NULL 
    END AS is_low_review,
    
    -- Order Details
    i.total_items,
    i.total_price,
    i.total_freight,
    i.total_weight_g

FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_customers_dataset.csv') c 
    ON o.customer_id = c.customer_id
LEFT JOIN items_agg i 
    ON o.order_id = i.order_id
LEFT JOIN reviews_clean r 
    ON o.order_id = r.order_id

-- Filter to successfully delivered orders with valid delivery dates
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_delivered_carrier_date IS NOT NULL
  AND o.order_delivered_customer_date >= o.order_purchase_timestamp;
