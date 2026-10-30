# Data Dictionary & Table Grain Notes

This document defines what a single row represents in each raw and engineered table within the Olist data pipeline, including primary keys, row counts, and grain definitions.

---

## 1. Raw Olist Relational Tables

| Table Name | Grain (What One Row Means) | Primary Key / Identifier | Row Count | Key Relationships & Foreign Keys |
| :--- | :--- | :--- | :---: | :--- |
| **`orders`** | **One row = one purchase order** placed by a customer on the marketplace. | `order_id` | 99,441 | Links to `customers.customer_id`. Contains lifecycle timestamps. |
| **`order_items`** | **One row = one physical item line in an order**. An order with 3 items has 3 rows. | `(order_id, order_item_id)` | 112,650 | Links to `orders.order_id`, `products.product_id`, and `sellers.seller_id`. |
| **`order_reviews`** | **One row = one customer review submission**. Mostly 1 per order, but 547 orders have $>1$ review. | `review_id` (or `order_id, review_id`) | 99,224 | Links to `orders.order_id`. Contains rating (1–5 stars) and optional survey comments. |
| **`customers`** | **One row = one customer transaction identity** tied to a specific order. | `customer_id` | 99,441 | `customer_id` is unique per order; `customer_unique_id` identifies recurring human buyers. |
| **`sellers`** | **One row = one independent merchant / seller** registered on Olist. | `seller_id` | 3,095 | Links to `order_items.seller_id`. Contains seller zip code prefix, city, and state. |
| **`products`** | **One row = one unique catalog product**. | `product_id` | 32,951 | Links to `order_items.product_id`. Contains physical weight (g), dimensions (cm), and category. |
| **`geolocation`** | **One row = one raw GPS coordinate observation** for a zip code prefix. Many rows per zip prefix. | None (raw GPS log) | 1,000,163 | Aggregated by `geolocation_zip_code_prefix` into spatial centroids. |
| **`order_payments`** | **One row = one payment installment or payment method transaction**. An order split across vouchers/cards has $>1$ row. | `(order_id, payment_sequential)` | 103,886 | Links to `orders.order_id`. Contains payment type, installments, and payment value. |
| **`product_category_name_translation`** | **One row = one category translation** mapping Portuguese names to English. | `product_category_name` | 71 | Reference translation table for `products.product_category_name`. |

---

## 2. Engineered Analytical Tables & Views

To enable regression modeling and prevent Cartesian row multiplication, raw tables are cleaned and aggregated into single-grain entities:

| Table / View | Grain (What One Row Means) | Primary Key | Row Count | Pipeline Role & Transformation Logic |
| :--- | :--- | :---: | :---: | :--- |
| **`geo_zip`** | **One row = one unique 5-digit zip code prefix** with mean centroid coordinates. | `zip_code_prefix` | 19,015 | Averages raw latitude and longitude per zip prefix to produce reliable geographic centroids (`sql/02_geo_clean.sql`). |
| **`reviews_clean`** | **One row = one unique order with its definitive review**. | `order_id` | 98,673 | Deduplicates multi-review orders using `QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_score ASC, review_answer_timestamp DESC) = 1` (`sql/03_reviews.sql`). |
| **`items_agg`** | **One row = one unique order with pre-aggregated item metrics**. | `order_id` | 98,666 | Computes `SUM(price)`, `SUM(freight)`, `SUM(weight)`, `COUNT(items)`, `COUNT(DISTINCT sellers)`, with `MIN()` main seller and category (`sql/04_items.sql`). |
| **`order_features`** | **One row = one delivered order** joined across customers, sellers, geo coordinates, items, and reviews. | `order_id` | 96,478 | Master join table computing Haversine distance, handling days, estimated transit days, and timing features (`sql/05_order_features.sql`). |
| **`seller_stats`** | **One row = one seller** with lifetime performance metrics and window function rankings. | `seller_id` | 3,095 | Computes total orders, late rate, avg review score, avg handling days, and rank for sellers with $\ge 30$ orders (`sql/06_seller_stats.sql`). |
| **`order_master`** | **One row = one clean delivered order ready for modeling** (Production Table). | `order_id` | 95,795 | Master analytical dataset after dropping 683 anomalous rows (0.71%), applying 99th percentile winsorization on weight, volume, and freight ratio, and adding log transforms (`sql/07_order_master_clean.sql`). |

---

## 3. Production Feature Dictionary (`order_master`)

Key fields used in econometric models (Models A and B):

| Column Name | Data Type | Formula / Description | Role in Modeling |
| :--- | :--- | :--- | :--- |
| `order_id` | VARCHAR | Unique order identifier. | Grain primary key (must match distinct count). |
| `is_late` | INTEGER | $1$ if `order_delivered_customer_date > order_estimated_delivery_date`, else $0$. | **Model A Target** / **Model B Predictor**. |
| `days_late` | DOUBLE | `order_delivered_customer_date - order_estimated_delivery_date` (in days). | Delay duration metric (negative if early). |
| `actual_days` | DOUBLE | `order_delivered_customer_date - order_purchase_timestamp` (in days). | Omitted from Model A (target leakage). |
| `estimated_days` | DOUBLE | `order_estimated_delivery_date - order_purchase_timestamp` (in days). | Promised delivery window buffer. |
| `seller_handling_days` | DOUBLE | `order_delivered_carrier_date - order_purchase_timestamp` (in days). | Warehouse dispatch time to carrier handoff. |
| `distance_km` | DOUBLE | Haversine great-circle distance between customer and seller centroids. | Geographic shipping distance friction. |
| `freight_ratio` | DOUBLE | `total_freight / total_price`. | Relative shipping fee burden. |
| `freight_ratio_capped` | DOUBLE | Winsorized at 99th percentile ($1.459$). | Skewness-controlled freight ratio predictor. |
| `total_weight_g` | HUGEINT | Sum of item weights per order. | Physical cargo mass. |
| `weight_capped` | DOUBLE | Winsorized at 99th percentile ($22,350$ g). | Skewness-controlled cargo mass. |
| `volume_capped` | DOUBLE | Winsorized at 99th percentile ($135,889$ cm³). | Skewness-controlled parcel cubic volume. |
| `same_state` | INTEGER | $1$ if `customer_state == seller_state`, else $0$. | Local intrastate fulfillment indicator. |
| `n_items` | BIGINT | Number of physical item units in the order. | Basket item count complexity. |
| `n_sellers` | BIGINT | Number of distinct sellers fulfilling the order. | Multi-seller basket fragmentation. |
| `review_score` | BIGINT | Customer rating from $1$ to $5$ stars. | Continuous review outcome (OLS). |
| `low_review` | INTEGER | $1$ if `review_score <= 2`, else $0$. | **Model B Binary Target** (Severe dissatisfaction). |
