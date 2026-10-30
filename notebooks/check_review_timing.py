import duckdb
import pandas as pd

con = duckdb.connect()

q = """
SELECT 
    CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END as is_late,
    CASE WHEN r.review_creation_date < o.order_delivered_customer_date THEN 1 ELSE 0 END as reviewed_before_delivery,
    r.review_score,
    COUNT(*) as count,
    ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0.0 END) * 100, 2) as pct_low_review
FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_order_reviews_dataset.csv') r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY is_late, reviewed_before_delivery, r.review_score
ORDER BY is_late, reviewed_before_delivery, r.review_score
"""

res = con.execute(q).df()
print(res)

q2 = """
SELECT 
    CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END as is_late,
    CASE WHEN r.review_creation_date < o.order_delivered_customer_date THEN 1 ELSE 0 END as reviewed_before_delivery,
    COUNT(*) as total_orders,
    SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) as low_reviews,
    ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0.0 END) * 100, 2) as pct_low_review
FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_order_reviews_dataset.csv') r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY is_late, reviewed_before_delivery
"""
res2 = con.execute(q2).df()
print("\n--- Summary by Delivery Timing and Review Timing ---")
print(res2)
