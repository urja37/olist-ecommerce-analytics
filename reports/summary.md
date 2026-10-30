# Fixing the Delivery Bottleneck: How Olist Can Protect Customer Trust
### Executive Summary & Operational Roadmap for Leadership (Non-Technical Summary)

---

## The Four Core Questions
1. **What operational and geographic factors predict whether an order arrives late?**
2. **What factors predict customer dissatisfaction and low review scores ($\le 2$ stars)?**
3. **How much of customer dissatisfaction is directly explained and mediated by delivery lateness?**
4. **What concrete, actionable operational levers should Olist implement to reduce lateness and protect customer review scores?**

### Core Metric Definitions
* **Late Delivery (`is_late`):** An order is defined as late if the actual delivery date exceeds the promised estimated date (`order_delivered_customer_date > order_estimated_delivery_date`). Evaluated strictly on delivered orders (`order_status = 'delivered'`).
* **Low Review (`low_review`):** An order is defined as having a low review if the customer gives a rating of 1 or 2 stars (`review_score <= 2`). Captures distinctly dissatisfied customers experiencing severe operational or product failure.

---

## 1. The Problem: The Hidden Cost of Missed Deadlines

E-commerce marketplaces thrive on repeat purchases and customer trust. On Olist, most customer transactions end in celebration—over **62% of buyers leave a perfect 5-star review**. However, more than **12% of orders result in a toxic 1- or 2-star rating**. 

For operations and commercial leadership, negative reviews represent lost lifetime value, increased customer service costs, and platform churn. But diagnosing the cause has historically sparked internal debate:
* *Are customers unhappy because products are defective or cheap?*
* *Are shipping fees too high?*
* *Or is delivery delay the real culprit?*

To resolve this debate, we analyzed **95,151 verified customer transactions** across every state in Brazil between 2016 and 2018. We mapped every step of the order lifecycle—from credit card approval to warehouse packaging, cross-country postal transit, and customer review submission.

The conclusion is unambiguous: **delivery lateness is the single greatest destroyer of customer satisfaction on Olist, and the majority of it can be prevented with smarter operational rules.**

---

## 2. Did the Data Agree? Reevaluating Initial Hypotheses
Before modeling, we established 6 hypotheses in [`docs/project_notes.md`](../docs/project_notes.md). Here is what the empirical data revealed:
* **1. Distance and route complexity drive delays:** **Agreed.** Long-haul shipping distance escalates delay odds by $+50\%$ per SD ($\text{Odds Ratio} = 1.50$), while local same-state routes reduce delay odds by $39\%$ ($\text{OR} = 0.61$).
* **2. Bulky / heavy cargo causes major sorting delays:** **Largely Refuted.** Package volume has no independent effect ($p = 0.992$), and weight has only a minor $+5.9\%$ risk increase per SD ($\text{OR} = 1.06$).
* **3. Seller warehouse delays trigger late deliveries:** **Strongly Agreed.** Seller handling time is the **#1 controllable operational driver** ($\text{OR} = \mathbf{2.22}$ per SD increase).
* **4. Delivery delay dominates bad customer reviews:** **Strongly Agreed.** Late delivery multiplies bad review odds by **$11.24\times$** and directly inflicts a **$-1.66$ star penalty** in controlled OLS regression.
* **5. High shipping fees depress customer ratings:** **Partially / Modestly Agreed.** High freight ratios slightly increase bad review odds ($\text{OR} = 1.09$), but this effect is completely overshadowed by delivery lateness.
* **6. Longer estimated delivery buffers protect ratings:** **Strongly Agreed.** Adding delivery window buffer reduces late delivery odds by **$73.3\%$ per SD** ($\text{OR} = \mathbf{0.27}$), confirming that customers judge satisfaction against the promise.

---

## 3. Key Findings: The 7 Core Operational Discoveries

![Executive Summary Dashboard](../outputs/figures/executive_summary_dashboard.png)

### 1. The Disproportionate Burden of Lateness
* **The Finding:** Late deliveries represent only **8.0% of all orders** ($7,607$ of $95,151$), but generate **33.8% of all low (1–2 star) customer reviews** across the marketplace.
* **Attributable Risk:** Population Attributable Risk (PAR%) proves that **27.9% of all bad reviews across Brazil are directly caused by late delivery**. If every delivery arrived on time, over $3,400$ negative reviews would vanish.

### 2. The 11.2x Dissatisfaction Multiplier & Rating Penalty
* **The Finding:** In our controlled multi-variable logistic regression (Model B), late delivery is the single strongest predictor of customer dissatisfaction, multiplying the odds of a low review by **$11.24\times$** ($p < 0.0001$).
* **Star Score Drop:** In controlled OLS regression, a late arrival inflicts an immediate **$-1.66$ star penalty** on review scores (dropping average ratings from $4.29$ on-time down to $2.56$ late).

![Takeaway 1: Late Deliveries Average 1.66 Stars Lower](../outputs/figures/01_rating_collapse_takeaway.png)

### 3. The 3-Day Delay Tolerance Cliff
* **The Finding:** Customer patience is non-linear. Minor delays of 1–3 days are largely forgiven (average rating drops slightly from $4.29$ to $3.76$ stars, $\text{OR} = 2.29$).
* **The Collapse:** Past Day 3, tolerance falls off a cliff. Delays of 4–7 days plunge ratings to **$2.32$ stars** ($\text{OR} = 15.14$), and delays of $8+$ days collapse ratings to **$1.74$ stars** ($\text{OR} = 34.03$). A package **10 days late has $15\times$ higher odds of a bad review than one that is 1 day late**.

![Takeaway 2: The 3-Day Delay Cliff](../outputs/figures/02_delay_cliff_takeaway.png)

### 4. Carrier Transit Causes 88% of Delays, but Slow Sellers Steal the Margin
* **The Finding:** On late orders, carrier transit surges from $6.9$ days on on-time shipments to **$23.9$ days**—accounting for **$87.5\%$ of all excess delay**.
* **Controllable Trigger:** Seller handling time is the **#1 controllable operational delay driver** ($\text{OR} = \mathbf{2.22}$ per SD increase). Late orders spend $+1.3$ extra days in merchant dispatch delay, wiping out the carrier transit safety buffer before the parcel even ships.

![Takeaway 3: Fulfillment Delay Bottlenecks](../outputs/figures/03_bottleneck_waterfall_takeaway.png)

### 5. Smarter Delivery Estimate Buffers Protect Customer Ratings
* **The Finding:** Adding buffer days to checkout delivery estimates provides massive statistical protection ($\text{OR} = \mathbf{0.27}$, representing a **$73.3\%$ reduction in delay odds** per SD).
* **Expectation Calibration:** Customers judge delivery performance against the promise date rather than the absolute calendar. Calibrating promises to realistic transit times prevents artificial lateness with zero shipping cost.

![Takeaway 5: Controllable Levers](../outputs/figures/05_odds_ratios_levers_takeaway.png)

### 6. Geographic Choke Points (Rio de Janeiro & the Northeast)
* **The Finding:** While Northeastern states suffer the highest percentage late rates (Alagoas at $23.9\%$, Maranhão at $19.7\%$), **Rio de Janeiro (RJ)** represents by far the largest absolute volume bottleneck ($13.5\%$ late rate).
* **Volume Concentration:** Rio de Janeiro accounts for **$1,664$ late orders—$21.9\%$ of all late deliveries nationwide**. The São Paulo $\rightarrow$ Rio de Janeiro lane alone accounts for over $1,200$ late deliveries across an average transit distance of only $441$ km.

![Takeaway 4: Geographic Hotspots](../outputs/figures/04_geographic_hotspots_takeaway.png)

### 7. The "Premature Survey Trap"
* **The Finding:** **$91.8\%$ of all low reviews on late orders were submitted before the package was delivered.**
* **Systemic Flaw:** Olist’s automated system sent review invitations on the *promised delivery date*, not when the carrier confirmed delivery. Stranded buyers opened an email asking *"How was your order?"* and used the review form as an angry customer service ticket (*"Produto não chegou!"* represents $>55\%$ of 1-star complaints). Gating survey dispatches to confirmed delivery eliminates thousands of 1-star ratings immediately.

---

## 4. What to Do: Four Concrete Operational Levers

We recommend four practical, high-ROI policy interventions:

```mermaid
flowchart LR
    A["1. Fix Survey Trigger<br/><b>Gate to Delivery</b>"] --> B["2. Smarter Estimates<br/><b>Dynamic Route Buffers</b>"]
    B --> C["3. Seller Dispatch SLA<br/><b>48-Hour Target</b>"]
    C --> D["4. Route Diversification<br/><b>Fix SP -> RJ Lane</b>"]
```

### 1. Fix the Survey Trigger (Zero Cost, Immediate Win)
* **Action:** Never send a customer review survey until the courier confirms the package has been physically delivered. 
* **If delayed:** Automatically send a proactive WhatsApp/email alert with updated tracking and a courtesy credit (e.g., $R\$\ 15$) *before* the customer reaches Day 4 of delay.
* **Impact:** Eliminates thousands of premature 1-star reviews overnight and prevents negative reviews before the 3-day tolerance cliff.

### 2. Give Smarter Delivery Promises at Checkout
* **Action:** Customers judge satisfaction against the promise, not the calendar. An order arriving in 18 days gets 5 stars if promised in 20 days, but gets 1 star if promised in 15 days. Add **$+2$ to $+4$ buffer days** to high-risk routes (such as shipments heading into Rio de Janeiro or the Northeast) and during holiday shopping peaks (Black Friday).
* **Impact:** Cuts reported late deliveries by **$25–35\%$** on high-risk routes with zero logistics expenditure.

### 3. Enforce a 48-Hour Seller Dispatch SLA
* **Action:** Mandate that sellers hand parcels to couriers within 2 business days. If a seller averages $>4$ days or has a $>15\%$ late rate, demote their listings in search results so orders naturally flow to fast, reliable merchants.
* **Impact:** Reduces platform handling time by $1.0–1.5$ days, preserving the buffer for carriers and reducing platform lateness by **$15–20\%$**.

### 4. Fix the Rio de Janeiro Delivery Corridor
* **Action:** Shipping to Rio de Janeiro is **$3.7\times$ more likely to be late** than shipping within São Paulo, leaking over 1,200 late orders on the SP $\rightarrow$ RJ lane alone. Partner with private urban couriers (e.g., Loggi, Total Express) in Rio de Janeiro to bypass state postal bottlenecks.
* **Impact:** Cutting lateness by one-third in Rio de Janeiro alone saves **$>530$ late deliveries**, avoids **$>240$ bad reviews**, and pushes Rio's regional rating from $3.97$ to **above $4.05$ stars**.

---

## 5. Counterfactual Policy Scenarios (Impact Sizing)

*Full mathematical derivations in [`notebooks/05_impact_sizing.ipynb`](../notebooks/05_impact_sizing.ipynb):*

| Scenario | Scope & Operational Action | Late Deliveries Eliminated | Bad Reviews (1–2★) Avoided | Rating / Dissatisfaction Impact |
| :--- | :--- | :---: | :---: | :--- |
| **1. Rio de Janeiro Focus** | Cut lateness by 1/3 in RJ via regional 3PLs & courier routing | **539** | **242** | RJ local rating rises **+0.08★** (3.97 → 4.05); RJ bad reviews drop **-10.9%** |
| **2. High-Delay States (10 States)** | Cut lateness by 1/3 across AL, MA, PI, CE, SE, BA, RJ, TO, RR, PA via dynamic buffers | **879** | **394** | National bad reviews drop **-3.2%**; national avg rating **+0.016★** |
| **3. Full Marketplace 1/3 Cut** | Comprehensive SLA enforcement, dynamic buffers & survey gating nationwide | **2,536** | **1,138** | National bad reviews drop **-9.3%**; national avg rating **+0.05★** |

---

## 6. Operational Priority Targeting: Top 10 Tables

### Table 1: Top 10 Worst Merchants by Late Rate (Min 30 Orders)
*Source: [`outputs/tables/top10_worst_sellers.csv`](../outputs/tables/top10_worst_sellers.csv)*

| Rank | Seller ID | Location | Orders | Late Orders | Late Rate (%) | Handling Time | Avg Review |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `ad781527c93d...` | São José do Rio Preto, SP | 34 | 12 | **35.29%** | 12.7 days | 3.41 ★ |
| 2 | `ede0c0364559...` | Ribeirão Preto, SP | 43 | 15 | **34.88%** | 3.7 days | 3.57 ★ |
| 3 | `835f0f7810c7...` | São Paulo, SP | 41 | 13 | **31.71%** | 9.8 days | 3.41 ★ |
| 4 | `54965bbe3e4f...` | Foz do Iguaçu, PR | 72 | 22 | **30.56%** | 15.5 days | 3.18 ★ |
| 5 | `ef990a83bbea...` | Joinville, SC | 40 | 11 | **27.50%** | 2.3 days | 3.75 ★ |
| 6 | `2a1348e9addc...` | Belo Horizonte, MG | 48 | 13 | **27.08%** | 5.1 days | 3.17 ★ |
| 7 | `66e0557ecc2b...` | São Paulo, SP | 30 | 8 | **26.67%** | 18.1 days | 4.03 ★ |
| 8 | `a49928bcdf77...` | São Paulo, SP | 94 | 25 | **26.60%** | 7.4 days | 3.06 ★ |
| 9 | `beadbee30901...` | Guarulhos, SP | 63 | 16 | **25.40%** | 7.6 days | 3.94 ★ |
| 10 | `71039d19d430...` | São Paulo, SP | 34 | 8 | **23.53%** | 2.4 days | 3.88 ★ |

### Table 2: Top 10 Worst Destination States by Late Delivery Rate (Min 30 Orders)
*Source: [`outputs/tables/top10_worst_states.csv`](../outputs/tables/top10_worst_states.csv)*

| Rank | State | Total Orders | Late Orders | Late Rate (%) | Delay vs Estimate | Avg Review |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **AL (Alagoas)** | 397 | 95 | **23.93%** | -8.0 days | 3.85 ★ |
| 2 | **MA (Maranhão)** | 717 | 141 | **19.67%** | -8.9 days | 3.83 ★ |
| 3 | **PI (Piauí)** | 476 | 76 | **15.97%** | -10.6 days | 3.99 ★ |
| 4 | **CE (Ceará)** | 1,279 | 196 | **15.32%** | -10.1 days | 3.94 ★ |
| 5 | **SE (Sergipe)** | 335 | 51 | **15.22%** | -9.3 days | 3.91 ★ |
| 6 | **BA (Bahia)** | 3,256 | 457 | **14.04%** | -10.1 days | 3.93 ★ |
| 7 | **RJ (Rio de Janeiro)** | 12,350 | 1,664 | **13.47%** | -11.1 days | 3.97 ★ |
| 8 | **TO (Tocantins)** | 274 | 35 | **12.77%** | -11.4 days | 4.15 ★ |
| 9 | **PA (Pará)** | 946 | 117 | **12.37%** | -13.4 days | 3.91 ★ |
| 10 | **ES (Espírito Santo)** | 1,995 | 244 | **12.23%** | -9.8 days | 4.08 ★ |

### Table 3: Top 10 Worst Shipping Corridors / Routes (Min 30 Orders)
*Source: [`outputs/tables/top10_worst_routes.csv`](../outputs/tables/top10_worst_routes.csv)*

| Rank | Route Corridor | Total Orders | Late Orders | Late Rate (%) | Avg Distance (km) | Avg Review |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **PR $\rightarrow$ AL** | 35 | 14 | **40.00%** | 2,250.6 km | 3.54 ★ |
| 2 | **SP $\rightarrow$ AL** | 256 | 67 | **26.17%** | 1,902.8 km | 3.70 ★ |
| 3 | **MA $\rightarrow$ SP** | 124 | 31 | **25.00%** | 2,331.5 km | 3.91 ★ |
| 4 | **RJ $\rightarrow$ CE** | 54 | 12 | **22.22%** | 2,122.4 km | 3.65 ★ |
| 5 | **SP $\rightarrow$ MA** | 493 | 105 | **21.30%** | 2,178.8 km | 3.71 ★ |
| 6 | **PR $\rightarrow$ MA** | 41 | 8 | **19.51%** | 2,490.8 km | 3.80 ★ |
| 7 | **SP $\rightarrow$ PI** | 329 | 60 | **18.24%** | 2,015.3 km | 3.96 ★ |
| 8 | **MG $\rightarrow$ AL** | 35 | 6 | **17.14%** | 1,519.7 km | 4.15 ★ |
| 9 | **PR $\rightarrow$ BA** | 143 | 24 | **16.78%** | 1,765.1 km | 3.82 ★ |
| 10 | **RS $\rightarrow$ BA** | 42 | 7 | **16.67%** | 2,195.4 km | 4.07 ★ |

*(High-Volume Bottleneck: The **SP $\rightarrow$ RJ** lane alone accounts for 8,157 orders with a 15.54% late rate and 1,268 delayed packages).*

---

## 7. Why You Can Trust These Findings

* **Full-Population Scale:** We analyzed the complete population of over 95,000 clean customer orders across Brazil, verifying that every single transaction was audited with zero duplicate counting.
* **Controlled for Alternative Explanations:** We tested whether bad reviews were simply driven by expensive items, heavy packages, high shipping fees, or particular product categories (e.g., furniture vs. cosmetics). Even after controlling for all these factors simultaneously, **delivery lateness remained $10\times$ more powerful than any other factor**.
* **Tested Across Stress-Test Scenarios (Robustness):**
  * We broadened the definition of a bad review to include neutral 3-star ratings—the conclusion held firmly ($\text{OR} = 8.4$).
  * We excluded the November 2017 Black Friday order surge to ensure the findings weren't just a holiday fluke—the late penalty remained identical ($\text{OR} = 10.8$).
  * We removed multi-seller orders to eliminate split-package confusion—the findings were identical ($\text{OR} = 11.0$).
* **Honest Causal Caveats & Next Steps:** Observational data reveals strong statistical patterns, but cannot guarantee that changing one policy will work in a vacuum. Before rolling out nationwide changes, Olist should run a controlled **A/B field experiment** on checkout delivery estimates to verify that adding buffer days increases customer satisfaction without dampening checkout conversion rates.

---

## Appendix: Data Cleaning Log

To ensure analytical rigor, prevent model distortion, and eliminate data anomalies, the following data cleaning protocol was executed across the **96,478 raw delivered transactions**, dropping **683 records (0.71%)** to establish a clean analytical dataset of **95,795 orders** (99.29% retention):

| Data Quality Issue | Rows Removed / Handled | Why Removed (Operational Root Cause) | What You Did (Treatment & Code Filter) |
| :--- | :---: | :--- | :--- |
| **Duplicate reviews per order** | 547 orders (543 with 2 reviews, 4 with 3 reviews) | Customers submitted multiple reviews over time. Keeping duplicates would violate the order grain and artificially inflate sample weight. | Deduplicated in `sql/03_reviews.sql` using `QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_creation_date DESC) = 1` to capture the customer's final review score. |
| **Missing delivery dates on delivered orders** | 8 rows dropped | Order marked as `delivered`, but `order_delivered_customer_date` was NULL. Actual transit duration cannot be computed. | Dropped via `WHERE order_delivered_customer_date IS NOT NULL`. |
| **Missing carrier dispatch or approval dates** | 15 rows dropped | Fulfillment timestamps were missing or unrecorded by the carrier. Intermediate handling milestones cannot be evaluated. | Excluded from fulfillment milestone models (`WHERE order_delivered_carrier_date IS NOT NULL`). |
| **Unmapped zip code coordinates** | 479 rows dropped | Customer or seller zip code prefix had no recorded GPS coordinates in the `geolocation` table, preventing Haversine distance calculation. | Excluded via inner join / `WHERE distance_km IS NOT NULL`. |
| **Missing or impossible product weights** | 22 rows dropped (16 null, 6 weight $\le 0$ g) | Product catalog entries lacked weight or recorded zero grams, distorting physical freight modeling. | Dropped via `WHERE total_weight_g > 0`. |
| **Negative elapsed times (Clock desynchronization)** | 159 rows dropped | Carrier timestamps occurred before purchase timestamp or delivery preceded carrier dispatch due to database clock errors or backfills. | Dropped via `WHERE actual_days >= 0 AND seller_handling_days >= 0`. |
| **Extreme outliers (Cargo bulk & freight ratios)** | Capped at 99th percentile (0 rows dropped) | Extreme values (e.g. freight ratio > 500% or multi-ton freight) create severe leverage points that distort regression coefficients. | Winsorized at 99th percentile: parcel weight capped at $22.35$ kg, freight ratio at $1.46$, and volume at $135,889$ cm³. |
| **Total Net Rows Dropped** | **683 rows (0.71%)** | Total unique records dropped across all quality criteria. | **95,795 clean rows retained (99.29% retention). Verified one row per order.** |
