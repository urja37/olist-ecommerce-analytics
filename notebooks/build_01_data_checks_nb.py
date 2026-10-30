import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title
cells.append(nbf.v4.new_markdown_cell("""# 01. Initial Data Sanity Checks & Database Audit
**Purpose:** Verify raw database tables, row counts, order status distributions, date ranges, and data integrity before downstream modeling.  
**Database:** `olist.duckdb`
"""))

# Cell 2: Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Connect to Embedded DuckDB Database"""))
cells.append(nbf.v4.new_code_cell("""import duckdb
import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

con = duckdb.connect("olist.duckdb")
print("Connected to olist.duckdb.")
"""))

# Cell 3: Row counts per table
cells.append(nbf.v4.new_markdown_cell("""## 2. Table Volumes & Row Counts
Verifying row counts across all 9 raw relational tables in the Olist database.
"""))
cells.append(nbf.v4.new_code_cell("""query_counts = \"\"\"
SELECT 'orders' AS table_name, COUNT(*) AS row_count FROM orders
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'reviews', COUNT(*) FROM reviews
UNION ALL SELECT 'customers', COUNT(*) FROM customers
UNION ALL SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'geo', COUNT(*) FROM geo
UNION ALL SELECT 'payments', COUNT(*) FROM payments
UNION ALL SELECT 'category_tr', COUNT(*) FROM category_tr;
\"\"\"
df_counts = con.execute(query_counts).df()
print("=== RAW TABLE ROW COUNTS ===")
display(df_counts)
"""))

# Cell 4: Order status counts
cells.append(nbf.v4.new_markdown_cell("""## 3. Order Status Distribution
Analyzing the status breakdown of orders. Note that only orders with `order_status = 'delivered'` represent completed delivery journeys.
"""))
cells.append(nbf.v4.new_code_cell("""query_status = \"\"\"
SELECT 
    order_status,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_share
FROM orders
GROUP BY order_status
ORDER BY order_count DESC;
\"\"\"
df_status = con.execute(query_status).df()
print("=== ORDER STATUS DISTRIBUTION ===")
display(df_status)
"""))

# Cell 5: Date range
cells.append(nbf.v4.new_markdown_cell("""## 4. Date Range Coverage
Auditing chronological coverage across purchase, delivery, and estimated delivery dates.
"""))
cells.append(nbf.v4.new_code_cell("""query_dates = \"\"\"
SELECT 
    MIN(order_purchase_timestamp) AS earliest_purchase,
    MAX(order_purchase_timestamp) AS latest_purchase,
    MIN(order_delivered_customer_date) AS earliest_delivery,
    MAX(order_delivered_customer_date) AS latest_delivery,
    MIN(order_estimated_delivery_date) AS earliest_estimated,
    MAX(order_estimated_delivery_date) AS latest_estimated
FROM orders;
\"\"\"
df_dates = con.execute(query_dates).df()
print("=== CHRONOLOGICAL COVERAGE ===")
display(df_dates)
"""))

# Cell 6: Multi-review orders and missing delivery dates
cells.append(nbf.v4.new_markdown_cell("""## 5. Preliminary Data Hygiene Checks
Identifying duplicate reviews, missing delivery dates on delivered orders, and logical timestamp anomalies.
"""))
cells.append(nbf.v4.new_code_cell("""# Orders with multiple reviews
query_multireview = \"\"\"
SELECT 
    review_count,
    COUNT(*) AS number_of_orders
FROM (
    SELECT order_id, COUNT(*) AS review_count
    FROM reviews
    GROUP BY order_id
    HAVING COUNT(*) > 1
) sub
GROUP BY review_count
ORDER BY review_count DESC;
\"\"\"
df_multirev = con.execute(query_multireview).df()
print("=== MULTI-REVIEW ORDERS BREAKDOWN ===")
display(df_multirev)

# Delivered orders with missing delivery dates
query_missing_del = \"\"\"
SELECT 
    COUNT(*) AS delivered_missing_delivery_date
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NULL;
\"\"\"
df_missing = con.execute(query_missing_del).df()
print(f"Delivered orders with missing customer delivery date: {df_missing.iloc[0,0]}")

# Negative elapsed times (carrier handoff before purchase)
query_neg = \"\"\"
SELECT COUNT(*) AS carrier_handoff_before_purchase
FROM orders
WHERE order_delivered_carrier_date < order_purchase_timestamp;
\"\"\"
df_neg = con.execute(query_neg).df()
print(f"Orders with carrier handoff before purchase: {df_neg.iloc[0,0]}")
"""))

# Cell 7: Summary & Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 6. Audit Summary & Downstream Takeaways
1. **Delivered Cohort:** 96,478 orders ($97.02\%$) are marked `delivered`, forming the primary cohort for delivery delay analysis.
2. **Date Range:** The transactions span ~25 months from September 2016 through October 2018.
3. **Data Cleaning Protocol:** 547 orders with multiple reviews require deduplication (`sql/03_reviews.sql`), and 8 delivered orders with null delivery dates must be excluded.
"""))

nb['cells'] = cells

target_nb = "notebooks/01_data_checks.ipynb"
with open(target_nb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Executing {target_nb}...")
client = NotebookClient(nb, timeout=300, kernel_name="python3")
client.execute()

with open(target_nb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully created and executed {target_nb} with rendered output cells!")
