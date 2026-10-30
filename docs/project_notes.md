# Project Notes & Research Hypotheses

This document records the initial research hypotheses formulated at project kickoff, operational notes on modeling trade-offs, and an assessment of whether the empirical data confirmed or refuted each hypothesis.

---

## 1. Initial Research Hypotheses (Formulated Prior to Modeling)

Prior to running regression models, the following hypotheses were established based on e-commerce operations theory:

1. **Hypothesis 1 (Distance Friction):**
   * *Claim:* Geographic transit distance and interstate route crossings are the primary drivers of delivery delays in Brazil.
2. **Hypothesis 2 (Cargo Weight & Dimensions):**
   * *Claim:* Heavy, bulky, or high-volume packages face severe handling friction at postal sorting facilities, significantly increasing late delivery odds.
3. **Hypothesis 3 (Seller Warehouse Dispatch Lag):**
   * *Claim:* Seller handling time (time from purchase approval to carrier handoff) directly eats into the delivery buffer and strongly predicts late delivery.
4. **Hypothesis 4 (Delivery Delay Drives Bad Reviews):**
   * *Claim:* Delivery lateness is the primary determinant of customer dissatisfaction, drastically increasing the probability of a 1- or 2-star review.
5. **Hypothesis 5 (Freight Burden Sensitivity):**
   * *Claim:* High shipping costs relative to item price (high freight ratio) make customers more critical and impatient, directly lowering review scores.
6. **Hypothesis 6 (Estimated Date Protection):**
   * *Claim:* Longer promised delivery windows at checkout protect against lateness without physically altering truck speeds.

---

## 2. Hypothesis Evaluation: Did the Data Agree?

| Hypothesis | Expected Effect | Empirical Finding (Model Evidence) | Verdict | Takeaway & Nuance |
| :--- | :--- | :--- | :---: | :--- |
| **H1: Distance Friction** | Higher distance $\rightarrow$ more delays | **Confirmed** (Model A: `log_distance` $\text{OR} = 1.50$, $p < 0.0001$; `same_state` $\text{OR} = 0.61$). | **Strongly Agreed** | Cross-country routes ($>1,000$ km) face nearly double the delay rate of local routes ($11.9\%$ vs. $6.4\%$). |
| **H2: Cargo Weight & Volume** | Heavy/bulky $\rightarrow$ more delays | **Partially Refuted** (Model A: `log_weight` $\text{OR} = 1.06$, $p = 0.015$; `log_volume` $\text{OR} = 1.00$, $p = 0.992$). | **Weak / Mixed** | After controlling for distance and handling time, volume has no independent effect. Weight has only a minor $+5.9\%$ risk increase per SD. |
| **H3: Seller Handling Lag** | Slow dispatch $\rightarrow$ more delays | **Confirmed** (Model A: `seller_handling_days` $\text{OR} = \mathbf{2.22}$, $p < 0.0001$). | **Strongly Agreed** | Seller handling time is the **#1 controllable operational driver** of delay; late orders spend $+1.26$ extra days in warehouse delay. |
| **H4: Delay Drives Bad Reviews** | Lateness $\rightarrow$ low reviews | **Confirmed** (Model B: `is_late` $\text{OR} = \mathbf{11.24}$, $p < 0.0001$; OLS drop: **$-1.66$ stars**). | **Strongly Agreed** | Lateness is the dominant driver of dissatisfaction. Furthermore, delays past 3 days cause ratings to collapse from $3.76$ to $2.32$ stars. |
| **H5: Freight Burden Sensitivity** | High freight ratio $\rightarrow$ low reviews | **Weak / Modest** (Model B: `freight_ratio_capped` $\text{OR} = 1.09$, $p < 0.0001$; Model A: $p = 0.792$). | **Partially Agreed** | Freight ratio slightly increases bad review odds, but its impact is completely overshadowed by delivery lateness ($\text{OR} = 11.24$). |
| **H6: Estimated Date Buffer** | Longer buffer $\rightarrow$ fewer delays | **Confirmed** (Model A: `estimated_days` $\text{OR} = \mathbf{0.27}$, $p < 0.0001$). | **Strongly Agreed** | Expanding the promised delivery estimate reduces late delivery odds by **$73.3\%$ per SD**, confirming expectations are judged relative to the promise. |

---

## 3. Modeling Decisions & Trade-Off Notes

1. **Grain Simplification (`MIN()` on multi-item orders):**
   * *Context:* $90.1\%$ of orders have only 1 item; $98.7\%$ have only 1 seller.
   * *Decision:* Used `MIN(seller_id)` and `MIN(category)` in `sql/04_items.sql` to avoid multi-row cartesian explosion.
   * *Robustness Check:* In Check 3, we dropped all multi-seller carts ($N = 93,922$) and re-estimated Model B. The late delivery odds ratio remained identical ($\text{OR} = 11.02$ vs. $10.93$), proving the simplification did not bias results.
2. **Exclusion of Target Leakage Features:**
   * Features `actual_days` (total elapsed delivery days) and `days_late` (days past estimated date) were deliberately excluded from Model A. Including them would represent tautological target leakage (predicting lateness using the delivery date itself).
3. **Winsorization at the 99th Percentile:**
   * Raw maximums for weight ($184.4$ kg) and freight ratio ($21.45$) reflected severe data anomalies and extreme outliers. Capping at 99th percentiles ($22.35$ kg and $1.46$) ensured stable logistic regression convergence without discarding valid orders.
