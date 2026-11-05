# Olist E-Commerce Operations Analytics & Delivery Optimization

> An end-to-end data analytics and econometric modeling project on ~100k Brazilian e-commerce orders, analyzing what operational and geographic factors cause late deliveries, quantifying their direct impact on customer review scores, and designing high-ROI policy interventions.

---

## The Question

This project addresses four core operational and business questions:

1. **What operational and geographic factors predict whether an order arrives late?**
2. **What factors predict a low customer review score ($\le 2$ stars)?**
3. **How much of customer dissatisfaction is directly explained and mediated by delivery lateness?**
4. **What concrete, actionable operational levers should Olist implement to reduce lateness and protect customer review scores?**

---

## Key Findings

### Headline Numbers (At a Glance)
* **8.0%** of deliveries arrive late ($7,607$ of $95,151$ delivered orders).
* **33.8%** of all 1–2 star reviews come from late deliveries ($4,114$ of $12,187$ low reviews).
* **4.29 stars vs. 2.56 stars:** On-time orders average **4.29 stars**, while late orders average **2.56 stars** (a raw gap of **-1.73 stars**; **-1.66 stars** in controlled regression).
* **9.3% vs. 54.0%:** Low review rate on-time is **9.3%**, but surges to **54.0%** when late ($5.8\times$ raw prevalence; $11.24\times$ logistic odds multiplier).
* **+0.08 stars / -11.3% low reviews:** Cutting lateness by a third in the worst states raises average review scores by **+0.08 points** and eliminates **1,370** bad reviews nationwide.

---

### Analytical Findings
* **The Disproportionate Burden of Lateness:** Late deliveries represent only **8.0% of all orders** ($7,607$ of $95,151$), but they generate **33.8% of all 1–2 star reviews** ($4,114$ of $12,187$). Population Attributable Risk (PAR%) demonstrates that **27.9% of all low reviews across the entire marketplace are directly caused by late delivery**.
* **The 11.2x Dissatisfaction Multiplier & Rating Penalty:** In our hierarchical logistic regression (Model B), late delivery is the single strongest predictor of customer dissatisfaction, multiplying the odds of a low review by **$11.24\times$** ($p < 0.0001$). In controlled OLS regression, late delivery inflicts a **$-1.66$ star penalty** on average (dropping ratings from $4.29$ on-time down to $2.56$ stars late).
* **The Non-Linear 3-Day Delay Cliff:** Minor delays of 1–3 days are largely tolerated by buyers ($-0.50$ stars, $\text{OR} = 2.29$), but delays of 4–7 days cause ratings to plunge by **$-1.91$ stars** ($\text{OR} = 15.14$), and delays of $8+$ days collapse ratings to $1.74$ stars ($\text{OR} = 34.03$). An order that is **10 days late has $15\times$ higher odds of a bad review than an order that is 1 day late**.
* **Seller Warehouse Lag is the #1 Controllable Cause:** Model A shows that seller handling time more than doubles the odds of late delivery per standard deviation increase ($\text{OR} = \mathbf{2.22}$, $p < 0.0001$). Bottom-quartile sellers take $>5$ days to dispatch, experiencing late rates $>25\%$. Conversely, expanding delivery estimate buffers provides massive protection ($\text{OR} = \mathbf{0.27}$, a **73.3% reduction in delay odds** per SD).
* **The "Premature Survey Trap":** **$91.8\%$ of all low reviews on late orders were submitted before the package was delivered**. Olist dispatched review invitation emails on the *estimated delivery date* rather than waiting for carrier delivery confirmation, prompting anxious customers to use the review form as an angry support ticket (*"Produto não chegou!"* represents $>55\%$ of 1-star complaints).

---

## Recommendations

1. **Set smarter delivery estimates:** Add +3 to +7 dynamic buffer days on chronically late routes (Rio de Janeiro, Northeast) to realign checkout promises with transit reality, eliminating 25–35% of late orders with zero logistics capex.
2. **Watch high-risk sellers:** Flag merchants averaging >4 days handling time or >15% late dispatch, coaching them on fulfillment and throttling Buy Box visibility for chronic offenders.
3. **Fix the worst routes:** Partner with regional 3PLs (e.g., Loggi in Rio de Janeiro) and pilot micro-fulfillment nodes in Greater RJ to bypass Correios bottlenecks on the high-volume SP → RJ corridor.
4. **Warn customers early:** Suppress automated review surveys until couriers confirm delivery, and send proactive WhatsApp/SMS alerts with R$ 15 apology vouchers before Day 3 of delay.

### Counterfactual Impact Sizing (Policy Scenarios)

| Scenario | Scope & Operational Action | Late Deliveries Eliminated | Bad Reviews (1–2★) Avoided | Rating / Dissatisfaction Impact |
| :--- | :--- | :---: | :---: | :--- |
| **1. Rio de Janeiro Focus** | Cut lateness by 1/3 in RJ via regional 3PLs & courier routing | **539** | **242** | RJ local rating rises **+0.08★** (3.97 → 4.05); RJ bad reviews drop **-10.9%** |
| **2. High-Delay States (10 States)** | Cut lateness by 1/3 across AL, MA, PI, CE, SE, BA, RJ, TO, RR, PA via dynamic buffers | **879** | **394** | National bad reviews drop **-3.2%**; national avg rating **+0.016★** |
| **3. Full Marketplace 1/3 Cut** | Comprehensive SLA enforcement, dynamic buffers & survey gating nationwide | **2,536** | **1,138** | National bad reviews drop **-9.3%**; national avg rating **+0.05★** |

*(Detailed recommendations with Driver, Evidence, Lever, and Result breakdown available in [`reports/recommendations.md`](reports/recommendations.md); full math computed in [`notebooks/05_impact_sizing.ipynb`](notebooks/05_impact_sizing.ipynb)).*

---

## Data and Method

### Core Metric Definitions
* **Late Delivery (`is_late`):** Defined as $1$ if `order_delivered_customer_date > order_estimated_delivery_date`, else $0$. Evaluated strictly on completed deliveries (`order_status = 'delivered'`).
* **Low Review (`low_review`):** Defined as $1$ if `review_score <= 2`, else $0$. Identifies distinctly dissatisfied buyers experiencing severe platform/fulfillment failure.

### 1. Raw Datasets Used
The analysis ingests all 9 relational tables from Olist's public dataset into DuckDB (`olist.duckdb`):
* `orders` ($99,441$ rows): Order lifecycle timestamps (purchase, approval, carrier pickup, customer delivery, estimated date).
* `order_items` ($112,650$ rows): Product, seller, price, and freight details per item.
* `order_reviews` ($99,224$ rows): Review scores ($1–5$), titles, comments, and survey timestamps.
* `customers` ($99,441$ rows): Customer location and state.
* `sellers` ($3,095$ rows): Merchant location and state.
* `products` ($32,951$ rows): Weight, dimensions, and category.
* `geolocation` ($1,000,163$ rows): Latitude/longitude coordinates by zip code prefix.
* `order_payments` ($103,886$ rows): Payment type, installments, and transaction value.
* `product_category_name_translation` ($71$ rows): Portuguese to English category mappings.

### 2. Analytical Data Pipeline (`order_master`)
The analysis table was engineered across seven SQL modules:
1. **Data Hygiene & Auditing (`sql/01_data_checks.sql`):** Filtered scope to delivered orders (`order_status = 'delivered'`).
2. **Geospatial Centroids (`sql/02_geo_clean.sql`):** Deduplicated and aggregated $1$M raw coordinates into $19,015$ unique zip-prefix centroids (`geo_zip`).
3. **Review Deduplication (`sql/03_reviews.sql`):** Resolved multi-review orders using `QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_score ASC, review_answer_timestamp DESC) = 1` into $98,673$ unique records (`reviews_clean`).
4. **Item Aggregation (`sql/04_items.sql`):** Pre-aggregated item quantities, weights, volumes, and pricing to the order level, selecting representative main sellers and categories via `MIN()` into $98,666$ records (`items_agg`).
5. **Master Join & Geodesic Distance (`sql/05_order_features.sql`):** Joined orders to customers, sellers, geo centroids, items, and reviews. Computed straight-line transit distance via the Haversine formula in DuckDB. Verified one row per order (`COUNT(*) = COUNT(DISTINCT order_id) = 96,478`).
6. **Seller Window Functions (`sql/06_seller_stats.sql`):** Ranked sellers ($\ge 30$ orders) by late rates, handling days, and route performance.
7. **Cleaning & Outlier Treatment (`sql/07_order_master_clean.sql` & `notebooks/clean_order_features.py`):**
   * Dropped $683$ anomalous records ($0.71\%$) missing coordinates, dates, or weights, leaving **$95,795$ clean rows** ($99.29\%$ retention).
   * **Grain Verification:** Verified one row per order (`COUNT(*) == COUNT(DISTINCT order_id)` = $95,795$; 0 duplicates).
   * **Winsorization:** Capped 99th percentiles on weight ($22.35$ kg), freight ratio ($1.46$), and volume ($135,889$ cm³).
   * **Transforms:** Log-transformed right-skewed variables ($\ln(1+x)$) and grouped rare states/categories into `'other'`.

### 3. Econometric & Statistical Models
* **Model A (Late Delivery Logistic Regression):**
  * Target: `is_late` ($1/0$). Sample: $N = 76,636$ (80% train split, stratified, seed 42).
  * Excluded target leakage variables (`actual_days`, `days_late`). Standardized continuous predictors ($z$-scores).
  * Multicollinearity test: Variance Inflation Factors (VIF) evaluated across all predictors; all $\le 5.15$ (far below the critical cutoff of 10).
  * Discrimination: Test ROC-AUC = **0.7740**, Test PR-AUC = **0.3145**.
  * Threshold Calibration: Calibrated using Youden's J statistic ($\tau = 0.086$), catching $67.2\%$ of late deliveries vs. $8.8\%$ at the naive $0.50$ default.
* **Model B (Customer Dissatisfaction Logistic & OLS):**
  * Target: `low_review` ($\le 2$ stars) and raw `review_score` ($1–5$). Sample: $N = 95,151$ clean reviewed orders.
  * Two-Step Hierarchical Design: Step 1 without lateness ($R^2 = 0.055$) vs. Step 2 with lateness ($R^2 = 0.157$, Likelihood Ratio $\chi^2 = 5,964.86, p < 0.0001$).
  * OLS on 1–5 review score: Demonstrates late orders score $-1.66$ stars lower ($p < 0.0001$), tripling explained score variance ($R^2 = 0.173$).
* **Robustness Checks:** Confirmed stability across 4 checks: redefining low reviews as $\le 3$ ($\text{OR} = 8.39$), non-linear delay duration buckets (`days_late`), single-seller subset ($\text{OR} = 11.02$), and excluding Black Friday November 2017 ($\text{OR} = 10.80$).

---

## How to Run It

### 1. Environment Setup & Installation
```bash
# Clone the repository
git clone https://github.com/urja37/olist-ecommerce-analytics.git
cd olist-ecommerce-analytics

# Create and activate Python virtual environment (Python 3.10+)
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Quick Start: Entry-Level 5-Minute Analysis
If you want a fast, accessible entry-level walkthrough (single clean SQL query + concise Jupyter notebook with 3 core charts, basic logistic regression, and practical recommendations):
```bash
# 1. Run the clean single-query SQL script (joins raw tables, milestones & lateness flags)
duckdb olist.duckdb < sql/simple_orders_query.sql

# 2. Open the simple 5-minute analysis notebook
jupyter notebook notebooks/simple_analysis.ipynb
```

### 2. Execution Order (Full Production Pipeline)
Execute the project sequentially in the following exact order:

#### Step 1: SQL Data Transformation Pipeline (DuckDB)
Processes raw CSVs from `data/` and builds clean analytical tables:
```bash
duckdb olist.duckdb < sql/01_data_checks.sql          # 1. Audit status counts, row counts & dates
duckdb olist.duckdb < sql/02_geo_clean.sql           # 2. Clean zip coordinates & calculate centroids
duckdb olist.duckdb < sql/03_reviews.sql             # 3. Resolve multi-review orders (deduplicate)
duckdb olist.duckdb < sql/04_items.sql               # 4. Aggregate items, freight & select main seller
duckdb olist.duckdb < sql/05_order_features.sql      # 5. Build order features & Haversine distance
duckdb olist.duckdb < sql/06_seller_stats.sql        # 6. Rank sellers & compute route metrics
duckdb olist.duckdb < sql/07_order_master_clean.sql  # 7. Build production master analytical dataset
```

#### Step 2: Numbered Analysis & Modeling Notebooks
Run the core numbered notebooks sequentially (via Jupyter or automated compiler scripts):
```bash
# 01. Initial Data Checks & Status Audits
python notebooks/build_01_data_checks_nb.py   # Compiles and executes notebooks/01_data_checks.ipynb

# 02. Exploratory Data Analysis (Full Suite of 7 EDA Charts)
python notebooks/build_02_eda_nb.py           # Compiles and executes notebooks/02_eda.ipynb

# 03. Model A — Predicting Late Delivery (Logistic Regression, VIF, ROC-AUC)
python notebooks/build_03_model_late_nb.py    # Compiles and executes notebooks/03_model_late.ipynb

# 04. Model B — Customer Review Degradation (Logit & Two-Step OLS on Review Scores)
python notebooks/build_04_model_reviews_nb.py # Compiles and executes notebooks/04_model_reviews.ipynb

# 05. Counterfactual Impact Sizing & Policy Scenarios
python notebooks/build_05_impact_sizing_nb.py # Compiles and executes notebooks/05_impact_sizing.ipynb
```

#### Step 3: Standalone Modeling Scripts & Master Notebook (Alternative Direct Run)
```bash
# Run standalone feature matrix preparation and model diagnostics
python notebooks/prepare_features.py          # Builds standardized train/test arrays
python notebooks/generate_requested_charts.py # Generates standalone PNGs in outputs/
python notebooks/run_model_a.py               # Fits Model A & saves odds ratios CSV
python notebooks/evaluate_model_diagnostics.py# Computes VIFs & ROC/PR curves
python notebooks/run_model_b.py               # Fits Model B Logit + OLS Step 1 & 2
python notebooks/run_robustness_checks.py     # Executes 4 sensitivity checks
python notebooks/size_impact.py               # Computes counterfactual math

# Master interactive end-to-end notebook
jupyter notebook notebooks/olist_end_to_end_analysis.ipynb
```

---

## Limitations

**Association, not causation.** All models in this project are observational regressions — they measure statistical association between predictors and outcomes, not confirmed causal pathways. The core finding is that late delivery is *strongly associated* with low review scores (OR = 11×, −1.66 stars), but this association could be partially explained by a common cause we did not measure. For example, a seller who ships slowly *and* packs items poorly would produce both lateness and bad reviews independently of each other. Without carrier-level data (route-specific SLA compliance, hub congestion logs, vehicle capacity), we cannot fully disentangle true carrier delay from seller-induced delays that persist into transit. To prove that *reducing lateness causes higher reviews*, Olist would need a controlled A/B test: randomly assign orders matching high-risk seller-route profiles to a "buffer-day intervention" arm and measure the resulting review score change against an unmodified control group. The robustness checks (three alternate specifications, monotone dose-response across delay buckets) substantially increase our confidence that the relationship is real and directional — but they do not substitute for experimental evidence. All recommended actions should therefore be piloted in controlled experiments before full marketplace deployment.
* **Omitted Variable Bias:** Lateness may partially proxy for unobserved seller diligence (careless packaging, defective goods, missing parts) or localized courier depot strikes.
* **Voluntary Reviewing & Sentiment Spillover:** Submitting a review is voluntary; easygoing buyers who experience minor delays may simply opt out of reviewing, while frustrated customers actively seek out the review form and penalize other aspects of the purchase ("horn effect").
* **The Survey Timing Artifact:** 91.8% of low reviews on late orders were submitted *before* delivery occurred. Review scores partially reflect distress over missing packages rather than post-consumption product quality.
* **Geodesic Distance Approximation:** Haversine distance measures straight-line distance, omitting actual highway kilometers traversed through sorting hubs.
* **Order-Level Aggregation Simplifications:** For multi-item orders (9.9% of orders) and multi-seller carts (1.3% of orders), representative seller and product category were selected using `MIN(seller_id)` and `MIN(product_category_name)`. Robustness checks confirmed that dropping multi-seller orders entirely yielded identical results (late odds ratio 11.02 vs. 10.93 full sample).
* **Need for A/B Testing:** Recommended interventions (especially dynamic delivery estimate buffers) must be validated via field A/B tests to monitor potential checkout conversion tradeoffs before full marketplace deployment.

---

## Repo Structure

```text
olist-project/
├── data/                                 # Raw Olist CSV files (ignored by git)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   └── product_category_name_translation.csv
├── sql/                                  # Modular DuckDB SQL transformation pipeline
│   ├── 01_data_checks.sql                # Data audit & status breakdown
│   ├── 02_geo_clean.sql                  # Geolocation deduplication & zip centroids
│   ├── 03_reviews.sql                    # Multi-review order resolution
│   ├── 04_items.sql                      # Order-level item aggregation & main seller
│   ├── 05_order_features.sql             # Master join, Haversine distance, timing features
│   ├── 06_seller_stats.sql               # Seller rankings, route stats & window functions
│   └── 07_order_master_clean.sql         # Production cleaned table with outlier capping
├── notebooks/                            # Python modeling, evaluation & analysis notebooks
│   ├── 01_data_checks.ipynb              # Executed initial table audits, status breakdown & dates
│   ├── 02_eda.ipynb                      # Executed full suite of 7 exploratory data analysis charts
│   ├── 03_model_late.ipynb               # Executed Model A logistic regression on is_late, VIF & ROC
│   ├── 04_model_reviews.ipynb            # Executed Model B Logit & OLS two-step on review scores
│   ├── 05_impact_sizing.ipynb            # Executed counterfactual scenario modeling & impact math
│   └── olist_end_to_end_analysis.ipynb   # Master executed notebook (all 13 sections)
├── reports/                              # Detailed business playbooks & operational summaries
│   ├── summary.md                        # 1-2 page non-technical executive summary for leadership
│   └── recommendations.md                # 4 core recommendations with Driver, Evidence, Lever & Result
├── outputs/                              # Model outputs, diagnostics & high-res visualizations
│   ├── figures/                          # Best 5 takeaway charts + executive summary dashboard
│   │   ├── 01_rating_collapse_takeaway.png
│   │   ├── 02_delay_cliff_takeaway.png
│   │   ├── 03_bottleneck_waterfall_takeaway.png
│   │   ├── 04_geographic_hotspots_takeaway.png
│   │   ├── 05_odds_ratios_levers_takeaway.png
│   │   └── executive_summary_dashboard.png
│   ├── tables/                           # Operational risk ranking tables (top 10 sellers, states, routes)
│   │   ├── top10_worst_sellers.csv       # Bottom 10 merchants by late delivery rate (min 30 orders)
│   │   ├── top10_worst_states.csv        # Top 10 destination states by late rate
│   │   └── top10_worst_routes.csv        # Top 10 logistics corridors by late rate (min 30 orders)
│   ├── model_a_odds_ratios.csv           # Model A coefficients, odds ratios & CIs
│   ├── model_a_vif.csv                   # Variance Inflation Factors table
│   ├── model_a_threshold_comparison.csv  # Confusion matrices across decision thresholds
│   ├── model_a_diagnostics.png           # 4-panel diagnostic plot (ROC, PR, CM, thresholds)
│   ├── model_b_logit_comparison.csv      # Model B Step 1 vs Step 2 logit comparison
│   ├── model_b_ols_coefficients.csv      # OLS review score regression coefficients
│   ├── model_b_step_comparison.png       # Model B mediation & rating penalty charts
│   ├── robustness_checks_summary.csv     # Robustness checks comparison table
│   ├── robustness_delay_buckets.csv      # Delay severity (days_late) odds & score drops
│   └── robustness_checks.png             # Multi-panel robustness visual summary
├── olist.duckdb                          # Embedded analytical DuckDB database
├── README.md                             # Project documentation & executive summary
└── requirements.txt                      # Python dependencies
```

---

## Dataset & Credits

The data analyzed in this project is the **Brazilian E-Commerce Public Dataset by Olist**, publicly released on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) under the **CC BY-NC-SA 4.0** license.

* **Publisher & Source:** [Olist](https://olist.com/) (Brazilian marketplace integration and merchant operations platform).
* **Dataset Scope:** 100,000 anonymized real commercial customer orders placed between October 2016 and October 2018 across all 27 Brazilian states.
* **Citation & Link:** Olist, *Brazilian E-Commerce Public Dataset by Olist*, Kaggle (2018). Available at: [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

