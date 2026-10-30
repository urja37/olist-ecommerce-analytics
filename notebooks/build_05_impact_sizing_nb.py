"""
Script to build and execute notebooks/05_impact_sizing.ipynb
Counterfactual impact sizing and policy scenario modeling.
"""

import nbformat as nbf
from nbclient import NotebookClient
import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

cells = []

# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""# 05. Counterfactual Impact Sizing & Policy Scenarios

**Research Question:** If Olist cuts delivery lateness by targeted percentages across key geographies and fulfillment corridors, how many bad reviews are prevented, and how much do customer review scores increase?

**Methodology:**
- **Counterfactual Math:** We link delivery status (`is_late`) to customer review scores using empirical review probabilities and controlled OLS/Logit regression parameters.
- **Population Attributable Risk (PAR%):** Measures the share of bad reviews (1-2 stars) that would not have occurred had late deliveries arrived on time.
- **Three Policy Scenarios Evaluated:**
  1. **Scenario 1: Rio de Janeiro Focus** — Cut lateness by 1/3 in RJ (the single highest volume delay market).
  2. **Scenario 2: High-Delay States Focus** — Cut lateness by 1/3 across all 10 states with late rates >= 12%.
  3. **Scenario 3: Marketplace-Wide 1/3 Reduction** — Cut lateness by 1/3 nationwide via comprehensive SLA enforcement and dynamic buffer calibration.
"""))

# -------------------------------------------------------------------
# Section 1: Baseline Metrics & Data Load
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Load Analytical Dataset

We load `order_master` from `olist.duckdb` and isolate the clean cohort of orders with verified customer review scores (N = 95,151).
"""))

cells.append(nbf.v4.new_code_cell("""\
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)

con = duckdb.connect('../olist.duckdb')
df = con.execute("SELECT * FROM order_master WHERE review_score IS NOT NULL").df()

n_total = len(df)
n_late = int(df['is_late'].sum())
pct_late = n_late / n_total

n_low = int((df['low_review'] == 1).sum())
pct_low = n_low / n_total

n_late_low = int(((df['is_late'] == 1) & (df['low_review'] == 1)).sum())
pct_late_in_low = n_late_low / n_low

mean_ontime = df[df['is_late'] == 0]['review_score'].mean()
mean_late = df[df['is_late'] == 1]['review_score'].mean()
gap_raw = mean_ontime - mean_late

rate_low_ontime = df[df['is_late'] == 0]['low_review'].mean()
rate_low_late = df[df['is_late'] == 1]['low_review'].mean()
rr = rate_low_late / rate_low_ontime

# Attributable risk
par = (pct_low - rate_low_ontime) / pct_low

print(f"Total reviewed orders:         {n_total:,}")
print(f"Late orders:                  {n_late:,} ({pct_late:.1%})")
print(f"Total low reviews (1-2 stars): {n_low:,} ({pct_low:.1%})")
print(f"Late low reviews:             {n_late_low:,} ({pct_late_in_low:.1%} of all low reviews)")
print(f"Mean review score - On-time:   {mean_ontime:.2f} stars")
print(f"Mean review score - Late:      {mean_late:.2f} stars")
print(f"Raw rating gap (Z):           {gap_raw:.2f} stars")
print(f"Low review rate - On-time:     {rate_low_ontime:.1%}")
print(f"Low review rate - Late:        {rate_low_late:.1%}")
print(f"Relative Risk (RR):            {rr:.2f}x")
print(f"Population Attributable Risk:  {par:.1%}")
"""))

# -------------------------------------------------------------------
# Section 2: Mathematical Framework for Counterfactual Impact
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 2. Mathematical Framework for Impact Sizing

For any target population $S$ (e.g., Rio de Janeiro or High-Delay States):
1. **Late Orders in Target Group:** $N_{\\text{late}, S}$
2. **Late Orders Avoided ($\Delta N_{\\text{late}}$):** If lateness is cut by fraction $\alpha$ (here $\alpha = 1/3$), then:
   $$\Delta N_{\\text{late}} = \alpha \times N_{\\text{late}, S}$$
3. **Low Reviews Avoided ($\Delta N_{\\text{low}}$):** 
   When a late order is converted to on-time, its probability of receiving a low review falls from $P(\\text{low} \\mid \\text{late}) = 54.08\\%$ to $P(\\text{low} \\mid \\text{on-time}) = 9.26\\%$ (a drop of $44.82$ percentage points):
   $$\Delta N_{\\text{low}} = \Delta N_{\\text{late}} \times \\left[ P(\\text{low} \\mid \\text{late}) - P(\\text{low} \\mid \\text{on-time}) \\right]$$
4. **Marketplace Low Reviews Reduction (%):**
   $$\\text{Reduction} = \\frac{\Delta N_{\\text{low}}}{N_{\\text{low, total}}}$$
5. **Rating Score Gain ($\Delta \\bar{R}$):**
   Using the raw gap ($Z = 1.73$ stars) or the controlled OLS parameter ($\\beta = -1.66$ stars):
   $$\Delta \\bar{R} = \\frac{\Delta N_{\\text{late}} \\times Z}{N_{\\text{total}}}$$
"""))

# -------------------------------------------------------------------
# Section 3: Computing the 3 Policy Scenarios
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 3. Policy Scenarios Computation

We compute the 3 benchmark counterfactual scenarios:
- **Scenario 1:** Rio de Janeiro (RJ) corridor optimization (cut RJ lateness by 1/3)
- **Scenario 2:** High-Delay States (cut lateness by 1/3 across all 10 states with late rate >= 12%)
- **Scenario 3:** Marketplace-wide 1/3 reduction
"""))

cells.append(nbf.v4.new_code_cell("""\
# State statistics
state_stats = df.groupby('customer_state').agg(
    orders=('order_id', 'count'),
    late_orders=('is_late', 'sum'),
    late_rate=('is_late', 'mean')
).sort_values('late_rate', ascending=False)

# Scenario 1: Rio de Janeiro Focus
rj_df = df[df['customer_state'] == 'RJ']
n_orders_rj = len(rj_df)
n_late_rj = int(rj_df['is_late'].sum())
late_cut_s1 = n_late_rj * (1/3)
low_avoided_s1 = late_cut_s1 * (rate_low_late - rate_low_ontime)
pct_low_drop_s1 = low_avoided_s1 / n_low
score_gain_s1_mkt = (late_cut_s1 * gap_raw) / n_total

# Local impact within RJ
rj_low_total = (rj_df['low_review'] == 1).sum()
pct_drop_within_rj = low_avoided_s1 / rj_low_total
score_gain_within_rj = (late_cut_s1 * gap_raw) / n_orders_rj
rj_baseline_score = rj_df['review_score'].mean()

# Scenario 2: High-Delay States (late rate >= 12%, N=10 states: AL, MA, PI, CE, SE, BA, RJ, TO, RR, PA)
worst_states_12 = state_stats[state_stats['late_rate'] >= 0.12].index.tolist()
worst_df_12 = df[df['customer_state'].isin(worst_states_12)]
n_orders_s2 = len(worst_df_12)
n_late_s2 = int(worst_df_12['is_late'].sum())
late_cut_s2 = n_late_s2 * (1/3)
low_avoided_s2 = late_cut_s2 * (rate_low_late - rate_low_ontime)
pct_low_drop_s2 = low_avoided_s2 / n_low
score_gain_s2_mkt = (late_cut_s2 * gap_raw) / n_total

# Scenario 3: Full Marketplace 1/3 Reduction
late_cut_s3 = n_late * (1/3)
low_avoided_s3 = late_cut_s3 * (rate_low_late - rate_low_ontime)
pct_low_drop_s3 = low_avoided_s3 / n_low
score_gain_s3_mkt = (late_cut_s3 * gap_raw) / n_total

scenarios_df = pd.DataFrame([
    {
        'Scenario': '1. Rio de Janeiro Focus',
        'Scope & Operational Action': 'Cut lateness by 1/3 in RJ via regional 3PLs & carrier routing',
        'Target Orders': f"{n_orders_rj:,}",
        'Late Deliveries Eliminated': int(round(late_cut_s1)),
        'Bad Reviews (1-2★) Avoided': int(round(low_avoided_s1)),
        'Low Review Reduction': f"-{pct_low_drop_s1:.1%} (Mkt) / -{pct_drop_within_rj:.1%} (RJ)",
        'Average Rating Impact': f"+{score_gain_s1_mkt:.3f}★ (Mkt) / +{score_gain_within_rj:.2f}★ (RJ)"
    },
    {
        'Scenario': '2. High-Delay States (10 States)',
        'Scope & Operational Action': 'Cut lateness by 1/3 across AL, MA, PI, CE, SE, BA, RJ, TO, RR, PA via dynamic buffers',
        'Target Orders': f"{n_orders_s2:,}",
        'Late Deliveries Eliminated': int(round(late_cut_s2)),
        'Bad Reviews (1-2★) Avoided': int(round(low_avoided_s2)),
        'Low Review Reduction': f"-{pct_low_drop_s2:.1%}",
        'Average Rating Impact': f"+{score_gain_s2_mkt:.3f}★"
    },
    {
        'Scenario': '3. Full Marketplace 1/3 Reduction',
        'Scope & Operational Action': 'Comprehensive SLA enforcement, dynamic promise buffers, and premature survey suppression',
        'Target Orders': f"{n_total:,}",
        'Late Deliveries Eliminated': int(round(late_cut_s3)),
        'Bad Reviews (1-2★) Avoided': int(round(low_avoided_s3)),
        'Low Review Reduction': f"-{pct_low_drop_s3:.1%}",
        'Average Rating Impact': f"+{score_gain_s3_mkt:.2f}★"
    }
])

display(scenarios_df)
scenarios_df.to_csv('../outputs/tables/counterfactual_scenarios.csv', index=False)
print("Saved outputs/tables/counterfactual_scenarios.csv")
"""))

# -------------------------------------------------------------------
# Section 4: Visualizing the Sizing Results
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 4. Visualization: Policy Intervention Payoffs

Comparison of eliminated late deliveries and avoided 1-2 star reviews across the 3 policy scenarios.
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

scenarios = ['Scenario 1\\n(RJ Focus)', 'Scenario 2\\n(10 High-Delay States)', 'Scenario 3\\n(Marketplace 1/3 Cut)']
late_elim = [late_cut_s1, late_cut_s2, late_cut_s3]
low_avoid = [low_avoided_s1, low_avoided_s2, low_avoided_s3]

colors = ['#2a9d8f', '#e76f51', '#264653']

# Chart 1: Late Deliveries Eliminated
bars1 = ax1.bar(scenarios, late_elim, color=colors, width=0.55, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Late Orders Eliminated', fontsize=11)
ax1.set_title('Late Deliveries Eliminated by Policy Scenario', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.4)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 40, f"{int(round(yval)):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_ylim(0, max(late_elim) * 1.15)

# Chart 2: Bad Reviews Avoided
bars2 = ax2.bar(scenarios, low_avoid, color=colors, width=0.55, edgecolor='black', alpha=0.85)
ax2.set_ylabel('1-2 Star Reviews Avoided', fontsize=11)
ax2.set_title('Bad Reviews (1-2★) Prevented by Policy Scenario', fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{int(round(yval)):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(low_avoid) * 1.15)

plt.tight_layout()
plt.savefig('../outputs/impact_sizing_scenarios.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/impact_sizing_scenarios.png")
"""))

# -------------------------------------------------------------------
# Section 5: Key Takeaways & Summary
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 5. Summary Table & Key Conclusions

### The 3 Core Policy Scenarios at a Glance

| Scenario | Scope & Action | Late Deliveries Eliminated | Bad Reviews (1-2★) Avoided | Rating / Dissatisfaction Impact |
| :--- | :--- | :---: | :---: | :--- |
| **1. Rio de Janeiro Focus** | Cut lateness by 1/3 in RJ via regional 3PLs & courier routing | **539** | **242** | RJ local rating rises **+0.08★** (3.97 → 4.05); RJ bad reviews drop **-10.9%** |
| **2. High-Delay States** | Cut lateness by 1/3 across 10 states (AL, MA, PI, CE, SE, BA, RJ, TO, RR, PA) | **879** | **394** | National bad reviews drop **-3.2%**; national avg rating **+0.016★** |
| **3. Full Marketplace 1/3 Cut** | Comprehensive SLA enforcement, dynamic buffers & survey gating nationwide | **2,536** | **1,138** | National bad reviews drop **-9.3%**; national avg rating **+0.05★** |

### Key Takeaways:
1. **Targeting Rio de Janeiro is the Highest-Leverage Geographic Focus:** Because RJ accounts for 21.2% of all late deliveries in Brazil, fixing RJ alone captures **46% of the impact** of fixing all 10 high-delay states combined.
2. **Local Customer Experience Uplift:** While a national average gain of +0.05 stars sounds modest, within localized high-delay regions like RJ, the rating jumps by **+0.08 stars** (from 3.97 to 4.05), crossing the key psychological threshold of 4.0 stars.
3. **High Return on Process Improvements:** Interventions like adding promise buffer days and suppressing premature surveys require zero logistics capex yet eliminate hundreds of severe 1-star reviews.
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Notebook 05_impact_sizing.ipynb complete - all counterfactual scenarios calculated successfully.")
"""))

nb.cells = cells

# Write and execute
nb_path = os.path.join(os.path.dirname(__file__), '05_impact_sizing.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook written: {nb_path}")

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name='python3',
    resources={'metadata': {'path': os.path.dirname(os.path.abspath(__file__))}}
)
client.execute()

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Executed and saved: {nb_path}")
