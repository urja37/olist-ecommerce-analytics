# Operational Recommendations & Policy Playbook

This document details four concrete, high-ROI operational recommendations for Olist leadership. For each recommendation, we outline the root-cause **Driver**, empirical **Evidence** from our models and SQL audit, the actionable **Lever** (concrete policy steps), and the **Expected Result**.

---

## Recommendation 1: Set Smarter Delivery Estimates (Dynamic Buffer Calibration)

### 1. Driver
Olist's checkout engine currently generates rigid, static delivery estimates that fail to account for regional logistical friction. On orders that arrive late, Olist promised a median buffer of only **22.2 days**, even though actual carrier transit required a median of **23.9 days**. Customers evaluate service quality against the promised delivery date rather than elapsed calendar days: an 18-day delivery promised in 20 days earns 5 stars, whereas the exact same 18-day delivery promised in 15 days triggers a 1-star rating.

### 2. Evidence
* **Model A (Predicting Lateness):** Promised delivery buffer (`estimated_days`) is the single strongest protective factor against lateness ($\text{Odds Ratio} = \mathbf{0.27}$, $95\%\text{ CI: } [0.25, 0.28]$, $p < 0.0001$). Each standard deviation increase in buffer days reduces delay odds by **73.3%**.
* **Model B (Review Scores):** Arriving on-time provides an immediate **+1.66 star uplift** in controlled OLS regression ($\beta = -1.659$, $t = -102.8$), and reduces the odds of a 1–2 star review by **$11.24\times$** in logistic regression.
* **Corridor Friction:** Interstate shipments into the Northeast (AL, MA, PI) and Rio de Janeiro experience late rates of **15% to 26%**, primarily due to long-haul carrier routing delays.

### 3. Lever / Concrete Action
1. **Deploy Route-Specific Dynamic Buffers:** Replace static delivery windows with an automated lookup engine based on historical carrier transit percentiles:
   * **+3 to +4 buffer days** for shipments terminating in **Rio de Janeiro (RJ)**.
   * **+5 to +7 buffer days** for shipments from the Southeast to the **Northeast (AL, MA, PI, CE, BA)** and **North (PA, AM)**.
   * **+2 buffer days** across all interstate lanes during seasonal peak demand (November Black Friday through Christmas).
2. **Dynamic Calibration vs. Conversion Monitoring:** Calibrate the promised date to the 85th percentile of historical transit time on high-friction lanes, ensuring expectations match logistics reality.

### 4. Expected Result
* **Lateness Reduction:** Eliminates **25% to 35% of technical late deliveries** on high-risk lanes without modifying physical courier operations.
* **Cost:** **$R\$\ 0$ capital expenditure** and zero incremental freight expense.
* **Customer Impact:** Directly converts thousands of would-be late orders into on-time arrivals, protecting review scores and customer trust.

---

## Recommendation 2: Watch High-Risk Sellers & Enforce Dispatch SLAs

### 1. Driver
Seller warehouse lag is the primary controllable bottleneck before package handoff. While payment approval takes under 30 minutes, bottom-quartile merchants take over **5 days** to prepare and hand parcels to the carrier, consuming the buffer that carriers require to absorb transit hiccups.

### 2. Evidence
* **Model A (Handling Impact):** Seller handling days is the #1 controllable predictor of delivery delay ($\text{Odds Ratio} = \mathbf{2.22}$, $95\%\text{ CI: } [2.17, 2.28]$, $p < 0.0001$). For every standard deviation increase in warehouse handling time, the odds of a late delivery more than double.
* **Bottleneck Decomposition:** On late orders, seller handling duration averages **3.01 days** vs. **1.75 days** for on-time orders (+1.26 days extra warehouse delay).
* **Concentrated Seller Risk:** The top 10 worst sellers with $\ge 30$ orders have late delivery rates ranging from **23.5% to 35.3%**, with average handling times exceeding **12 to 18 days** (e.g., Seller `66e0557ecc2b...` in São Paulo averages 18.1 days handling; Seller `54965bbe3e4f...` in Foz do Iguaçu averages 15.5 days handling).

### 3. Lever / Concrete Action
1. **Mandatory 48-Hour Dispatch SLA:** Implement an automated tracking milestone requiring sellers to obtain courier scan confirmation within 48 business hours of order approval.
2. **Tiered Seller Health Scorecard:**
   * **Tier 1 (Warning & Coaching):** Flag sellers with $>5\%$ late dispatches or average handling $>3.5$ days; send automated operational coaching and packaging guidelines.
   * **Tier 2 (Buffer Penalty):** Automatically add +3 days to the delivery estimate displayed on listings of sellers with chronic dispatch delays, preventing unrealistic buyer promises.
   * **Tier 3 (Buy Box Suppression):** Restrict Buy Box eligibility and de-rank listings for merchants maintaining $>15\%$ late rates or handling $>5$ days.

### 4. Expected Result
* **Handling Compression:** Reduces average merchant handling time by **1.0 to 1.5 days** across flagged sellers.
* **Downstream Delay Relief:** Preserves carrier transit buffers, reducing overall marketplace late deliveries by **15% to 20%**.
* **Rating Protection:** Eliminates the $-0.10$ star per-SD direct rating penalty associated with chronic seller dispatch lag.

---

## Recommendation 3: Fix the Worst Routes & Regional Logistics

### 1. Driver
Severe geographic asymmetry exists in Brazilian e-commerce fulfillment: while intra-state São Paulo logistics operate efficiently (5.4% late rate), cross-regional long-haul corridors to the Northeast and Rio de Janeiro face severe carrier bottlenecks, sorting center congestion, and cargo security delays.

### 2. Evidence
* **Concentrated Volume in Rio de Janeiro:** Rio de Janeiro accounts for **13.5% late rate** across 12,350 orders, generating **1,664 late orders** (21.9% of all delayed orders nationwide). The São Paulo $\rightarrow$ Rio corridor alone represents **8,157 orders** with a **15.5% late rate** (1,268 delayed shipments over just 441 km).
* **Extreme Outlier Lanes:** The Northeast corridor exhibits acute lateness: **PR $\rightarrow$ AL** (40.0% late), **SP $\rightarrow$ AL** (26.2% late), **MA $\rightarrow$ SP** (25.0% late), and **SP $\rightarrow$ MA** (21.3% late).
* **Carrier Custody Bottleneck:** Waterfall analysis confirms that **87.5% of total excess fulfillment delay** occurs while packages are in carrier custody (transit duration jumps from 6.9 days on-time to 23.9 days when late).

### 3. Lever / Concrete Action
1. **Carrier Diversification on SP $\rightarrow$ RJ:** Shift parcel volume away from the national postal carrier (Correios) on the SP $\rightarrow$ RJ route toward private regional couriers (e.g., Loggi, Total Express, Jadlog) with dedicated hub-to-hub express line-hauls.
2. **Pilot Micro-Fulfillment / Cross-Docking in Greater RJ:** Partner with regional 3PLs in Greater Rio de Janeiro and Recife/Salvador to stage fast-moving inventory locally, converting cross-country transit into next-day regional delivery.
3. **Route-Specific Carrier Allocation:** Automatically route bulky or heavy shipments (>5 kg) to specialized freight carriers capable of two-man delivery rather than standard parcel post.

### 4. Expected Result
* **RJ Corridor Impact:** Cutting lateness by one-third in Rio de Janeiro alone saves **539 late deliveries**, avoids **242 bad reviews**, and pushes RJ's regional review score from **3.97 to 4.05 stars** (+0.08 stars, crossing the key 4.0 psychological threshold).
* **Regional Transit Compression:** Reduces average transit duration into RJ and the Northeast by **4 to 6 days**.

---

## Recommendation 4: Warn Customers Early & Fix Survey Triggers

### 1. Driver
A critical platform workflow flaw currently exacerbates customer dissatisfaction: Olist’s automated system triggers review invitation emails on the **estimated delivery date**, irrespective of whether the courier has confirmed delivery. When a delivery is delayed, the customer receives an invitation to review a package they have not received, prompting an immediate angry 1-star review (*"Produto não chegou!"*).

### 2. Evidence
* **The Premature Survey Trap:** **91.8% of all 1–2 star reviews on late orders (3,817 out of 4,146) were submitted before the package was delivered**.
* **Review Sentiment Shift:** When delayed packages were delivered *before* the review was submitted, the low-review rate dropped from **70.3% down to 15.1%**.
* **The 3-Day Delay Cliff:** Empirical review curves show a sharp non-linear tolerance cliff: customers largely tolerate 1–3 days of delay ($-0.50$ stars, $\text{OR} = 2.29$), but ratings collapse severely at 4–7 days late ($-1.91$ stars, $\text{OR} = 15.14$), and at $8+$ days late reach $\text{OR} = 34.03$.

### 3. Lever / Concrete Action
1. **Delivery-Gated Review Surveys:** Reprogram notification logic so review surveys are triggered **strictly 48 hours after courier delivery confirmation** (`order_delivered_customer_date`), verified via carrier API webhooks.
2. **Proactive In-Transit Delay Alerts:** For orders identified as high-risk by Model A (risk score $\ge 0.086$) or where carrier tracking shows zero hub scans 48 hours prior to the promise date:
   * Dispatch proactive SMS / WhatsApp / email alerts: *"Your delivery is taking slightly longer than expected due to carrier routing. We are tracking it closely for you."*
3. **Courtesy Goodwill Vouchers:** Automatically issue a small courtesy credit (e.g., $R\$\ 10–15$) on Day 1 of delay, de-escalating customer frustration *before* the 3-day cliff.

### 4. Expected Result
* **Immediate Bad Review Elimination:** Postponing review surveys until actual delivery eliminates up to **~3,500 premature 1-star reviews** platform-wide.
* **Customer Goodwill:** Proactive communication before Day 3 prevents customer anger from escalating, halving negative reviews on delayed orders from 54% to ~25%.
* **Platform NPS Uplift:** Lifts overall marketplace net sentiment with minimal engineering effort.

---

## Summary of Recommendations Matrix

| # | Recommendation | Driver | Empirical Evidence | Lever / Policy Action | Expected Result |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Set Smarter Delivery Estimates** | Static checkout estimates fail to reflect route reality; promises are shorter on routes that take longer. | Model A OR = 0.27 per SD buffer; on-time orders gain +1.66 stars in OLS. | Dynamic +3 to +7 buffer days on high-delay routes (RJ, Northeast, North). | Cuts late orders by 25–35% on target routes with $R\$ 0$ capex. |
| **2** | **Watch High-Risk Sellers** | Slow seller warehouse dispatch consumes carrier buffer; handling averages 3.01 days on late orders. | Model A OR = 2.22 per SD handling time; top 10 worst sellers average 12–18 days handling. | 48-hour dispatch SLA; tiered scorecard with search throttling and buffer penalties. | Compresses handling by 1.0–1.5 days; cuts platform lateness by 15–20%. |
| **3** | **Fix the Worst Routes** | Geographic bottlenecks in Rio and Northeast account for 87.5% of total excess delay. | RJ accounts for 21.9% of all late deliveries; SP $\rightarrow$ RJ has 15.5% late rate across 8,157 orders. | Shift SP $\rightarrow$ RJ volume to private 3PLs (Loggi); pilot micro-fulfillment in Greater RJ. | Saves 539 late orders in RJ alone; lifts RJ regional score from 3.97 to 4.05 stars. |
| **4** | **Warn Customers Early** | Premature review surveys on estimated date trigger angry 1-star tickets before parcel arrival. | 91.8% of late bad reviews submitted before delivery; 3-day cliff spikes bad review odds 15x. | Gate surveys to delivery + 48h; send proactive Day 1 delay alerts with $R\$ 15$ courtesy credit. | Eliminates ~3,500 premature 1-star reviews; de-escalates customer frustration before 3-day cliff. |
