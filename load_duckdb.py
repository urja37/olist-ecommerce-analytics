import duckdb
import time

print("Connecting to olist.duckdb...")
start_time = time.time()
con = duckdb.connect("olist.duckdb")

tables = {
  "orders": "olist_orders_dataset.csv",
  "order_items": "olist_order_items_dataset.csv",
  "reviews": "olist_order_reviews_dataset.csv",
  "customers": "olist_customers_dataset.csv",
  "sellers": "olist_sellers_dataset.csv",
  "products": "olist_products_dataset.csv",
  "geo": "olist_geolocation_dataset.csv",
  "payments": "olist_order_payments_dataset.csv",
  "category_tr": "product_category_name_translation.csv",
}

print("Loading CSVs into DuckDB tables...")
for name, file in tables.items():
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('data/{file}')")
    cnt = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    elapsed = time.time() - t0
    print(f"  - Table '{name}' created: {cnt:,} rows ({elapsed:.2f}s)")

total_elapsed = time.time() - start_time
print(f"\nAll 9 tables loaded successfully in {total_elapsed:.2f}s!")

print("\nDatabase inspection:")
db_tables = con.execute("SHOW TABLES").fetchall()
for t in db_tables:
    table_name = t[0]
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    cols = con.execute(f"SELECT COUNT(*) FROM pragma_table_info('{table_name}')").fetchone()[0]
    print(f"  * {table_name}: {count:,} rows, {cols} columns")

con.close()
