import duckdb
import pandas as pd

con = duckdb.connect()

tables = [
    'olist_customers_dataset',
    'olist_geolocation_dataset',
    'olist_order_items_dataset',
    'olist_order_payments_dataset',
    'olist_order_reviews_dataset',
    'olist_orders_dataset',
    'olist_products_dataset',
    'olist_sellers_dataset',
    'product_category_name_translation'
]

print("=== DATASET ROW COUNTS ===")
for t in tables:
    path = f"data/{t}.csv"
    res = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}')").fetchone()[0]
    print(f"{t}: {res:,} rows")

print("\n=== ORDERS STATUS SUMMARY ===")
orders_summary = con.execute("""
    SELECT 
        order_status,
        COUNT(*) as cnt,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
    FROM read_csv_auto('data/olist_orders_dataset.csv')
    GROUP BY order_status
    ORDER BY cnt DESC
""").df()
print(orders_summary)

print("\n=== ORDERS DATE RANGE ===")
date_range = con.execute("""
    SELECT 
        MIN(order_purchase_timestamp) as min_purchase,
        MAX(order_purchase_timestamp) as max_purchase,
        MIN(order_delivered_customer_date) as min_delivery,
        MAX(order_delivered_customer_date) as max_delivery
    FROM read_csv_auto('data/olist_orders_dataset.csv')
    WHERE order_status = 'delivered'
""").df()
print(date_range)

print("\n=== LATENESS RATE IN DELIVERED ORDERS ===")
late_summary = con.execute("""
    SELECT 
        CASE 
            WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 
            ELSE 0 
        END as is_late,
        COUNT(*) as cnt,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
    FROM read_csv_auto('data/olist_orders_dataset.csv')
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
      AND order_estimated_delivery_date IS NOT NULL
    GROUP BY is_late
""").df()
print(late_summary)

print("\n=== REVIEW SCORES DISTRIBUTION ===")
reviews_summary = con.execute("""
    SELECT 
        review_score,
        COUNT(*) as cnt,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct,
        CASE WHEN review_score <= 2 THEN 1 ELSE 0 END as is_low_review
    FROM read_csv_auto('data/olist_order_reviews_dataset.csv')
    GROUP BY review_score
    ORDER BY review_score
""").df()
print(reviews_summary)
