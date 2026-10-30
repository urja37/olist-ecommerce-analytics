import duckdb

con = duckdb.connect()

print("=== CHECKING COLUMNS OF ALL TABLES ===")
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

for t in tables:
    cols = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('data/{t}.csv')").fetchall()
    print(f"\n--- {t} ---")
    for c in cols:
        print(f"  {c[0]} ({c[1]})")
