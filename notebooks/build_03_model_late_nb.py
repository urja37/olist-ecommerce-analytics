"""
Script to build and execute notebooks/03_model_late.ipynb
Model A: Logistic Regression — Predicting Late Delivery
"""

import nbformat as nbf
from nbclient import NotebookClient
import os

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
cells.append(nbf.v4.new_markdown_cell("""# 03. Model A — Predicting Late Delivery (Logistic Regression)

**Research question:** Which order-level factors best predict whether a delivery will arrive late?

**Method:** Logistic regression on 95,795 cleaned orders (80/20 train/test split, stratified on `is_late`).  
**Target:** `is_late` (1 if actual delivery > estimated delivery date, else 0; base rate ≈ 8.1 %).

**Key outputs:**
- Fitted coefficients with odds ratios and 95% CIs
- Variance Inflation Factors (VIF) — multicollinearity check
- Out-of-sample AUC and confusion matrix
- Robustness checks: alternate lateness definitions, sub-samples
"""))

# -------------------------------------------------------------------
# Section 1 — Setup & Data
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Load Clean Data

We load `order_master` — the pre-cleaned analytical table built by `sql/07_order_master_clean.sql`.
All cleaning decisions (683 rows dropped, 99.29 % retention) are documented in the Appendix of
`olist_operations_executive_report.md`.
"""))

cells.append(nbf.v4.new_code_cell("""\
import duckdb
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve)

# Reproduce environment used to generate saved outputs
import os
os.makedirs('../outputs', exist_ok=True)

con = duckdb.connect('../olist.duckdb')
df = con.execute('SELECT * FROM order_master').df()

print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"Late rate: {df['is_late'].mean():.2%}")
ts_min = pd.to_datetime(df['order_purchase_timestamp'].min())
ts_max = pd.to_datetime(df['order_purchase_timestamp'].max())
print(f"Date range: {ts_min.strftime('%Y-%m-%d')} → {ts_max.strftime('%Y-%m-%d')}")
df[['is_late','seller_handling_days','estimated_days','distance_km',
    'review_score','low_review']].describe().round(3)
"""))

# -------------------------------------------------------------------
# Section 2 — Feature Engineering
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Engineering

We create the feature matrix used in Model A.

**Continuous predictors** — standardised (zero mean, unit variance) so coefficients are
comparable across scales:
- `log_distance` — log(1 + Haversine km) — captures shipping distance non-linearly
- `log_weight`, `log_volume` — parcel size, log-scaled
- `n_items`, `n_sellers` — order complexity
- `freight_ratio_capped` — freight / price, capped at 99th percentile
- `seller_handling_days` — days from order approval to carrier pickup (strongest predictor)
- `estimated_days` — seller's promised window (longer buffer = lower late risk)
- `purchase_month` — seasonal demand patterns

**Categorical dummies** (drop-first to avoid perfect collinearity):
- Customer state (top 10 by volume, remainder → `other`)
- Product category (top 12 by volume, remainder → `other`)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Group high-cardinality states & categories
top_cust_states = df['customer_state'].value_counts().nlargest(10).index
df['cust_state_grouped'] = df['customer_state'].apply(
    lambda x: x if x in top_cust_states else 'other')

top_categories = df['category_en'].value_counts().nlargest(12).index
df['category_grouped'] = df['category_en'].apply(
    lambda x: x if x in top_categories else 'other')

continuous_cols = [
    'log_distance', 'log_weight', 'log_volume',
    'n_items', 'n_sellers', 'freight_ratio_capped',
    'seller_handling_days', 'estimated_days', 'purchase_month'
]

dummies = pd.get_dummies(
    df[['cust_state_grouped', 'category_grouped']],
    prefix=['cust_state', 'cat'], drop_first=True, dtype=float)

X_all = pd.concat([df[continuous_cols + ['same_state']], dummies], axis=1)
y_all = df['is_late']

# 80/20 train/test split — stratified on is_late (rare positive class)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# Standardise continuous predictors (fit on train only)
scaler = StandardScaler()
X_train = X_train.copy(); X_test = X_test.copy()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols]  = scaler.transform(X_test[continuous_cols])

print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
print(f"Train late rate: {y_train.mean():.2%}  |  Test late rate: {y_test.mean():.2%}")
print(f"Features: {X_train.shape[1]}")
"""))

# -------------------------------------------------------------------
# Section 3 — Fit Model A
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 3. Fit Logistic Regression (Model A)

Statsmodels `Logit` gives us exact p-values and 95% confidence intervals via MLE.
All 32 features are included simultaneously (not stepwise) to control for confounders.
"""))

cells.append(nbf.v4.new_code_cell("""\
X_train_c = sm.add_constant(X_train)
X_test_c  = sm.add_constant(X_test)

print("Fitting Model A … (takes ~30 s)")
model_a = sm.Logit(y_train, X_train_c).fit(maxiter=200)
print(model_a.summary())
"""))

# -------------------------------------------------------------------
# Section 4 — Odds Ratios & CIs
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 4. Odds Ratios & 95 % Confidence Intervals

An odds ratio (OR) > 1 means the feature *increases* the odds of late delivery.  
OR < 1 means it *decreases* the odds.  
We exponentiate the log-odds coefficients and their CIs.
"""))

cells.append(nbf.v4.new_code_cell("""\
conf = model_a.conf_int()
odds_df = pd.DataFrame({
    'Feature':        model_a.params.index,
    'Beta':           model_a.params.values,
    'SE':             model_a.bse.values,
    'z':              model_a.tvalues.values,
    'p_value':        model_a.pvalues.values,
    'OR':             np.exp(model_a.params.values),
    'CI_lower_95':    np.exp(conf[0].values),
    'CI_upper_95':    np.exp(conf[1].values),
}).query("Feature != 'const'").sort_values('OR', ascending=False)

print("=== TOP RISK FACTORS (OR > 1) ===")
display(odds_df[odds_df['OR'] >= 1].head(8)
        [['Feature','Beta','OR','CI_lower_95','CI_upper_95','p_value']].round(4))

print("\\n=== TOP PROTECTIVE FACTORS (OR < 1) ===")
display(odds_df[odds_df['OR'] < 1].tail(8)
        [['Feature','Beta','OR','CI_lower_95','CI_upper_95','p_value']].round(4))
"""))

# Forest plot
cells.append(nbf.v4.new_markdown_cell("""### 4.1 Forest Plot — Top 12 Odds Ratios

Only features with p < 0.05 shown. Error bars = 95 % CI.
"""))

cells.append(nbf.v4.new_code_cell("""\
sig = odds_df[odds_df['p_value'] < 0.05].copy()

# Show top 6 risk + top 6 protective
top_risk = sig[sig['OR'] >= 1].head(6)
top_prot = sig[sig['OR'] <  1].tail(6)
plot_df  = pd.concat([top_risk, top_prot]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = range(len(plot_df))
ax.errorbar(plot_df['OR'], y_pos,
            xerr=[plot_df['OR']-plot_df['CI_lower_95'],
                  plot_df['CI_upper_95']-plot_df['OR']],
            fmt='o', color='steelblue', ecolor='gray',
            elinewidth=1.5, capsize=4, markersize=7)
ax.axvline(1.0, color='red', linestyle='--', linewidth=1.2, label='OR = 1 (no effect)')
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['Feature'], fontsize=10)
ax.set_xlabel('Odds Ratio (95 % CI)', fontsize=11)
ax.set_title('Model A — Significant Predictors of Late Delivery\\n'
             'Seller handling time doubles the odds; buffer days cut them by 73 %', fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/model_a_forest_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/model_a_forest_plot.png")
"""))

# -------------------------------------------------------------------
# Section 5 — VIF
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 5. Multicollinearity Check — Variance Inflation Factors (VIF)

VIF < 5 is generally acceptable; VIF > 10 signals serious collinearity.  
We load the pre-computed VIF CSV (computed on the full feature matrix before splitting).
"""))

cells.append(nbf.v4.new_code_cell("""\
vif_df = pd.read_csv('../outputs/model_a_vif.csv').sort_values('VIF', ascending=False)
print(f"Max VIF: {vif_df['VIF'].max():.2f}  (threshold = 5)")
print(f"Features with VIF > 5: {(vif_df['VIF'] > 5).sum()}")
print()
display(vif_df.head(10).round(2))

# Flag
high_vif = vif_df[vif_df['VIF'] > 5]
if len(high_vif):
    print(f"\\n⚠ High VIF features: {list(high_vif['Feature'])}")
    print("Note: cust_state_SP is the largest state (≈42 % of orders); some collinearity\\n"
          "with log_distance and same_state is expected but coefficients remain interpretable.")
else:
    print("\\n✓ All VIFs below 5 — no problematic multicollinearity.")
"""))

# -------------------------------------------------------------------
# Section 6 — Out-of-Sample Evaluation
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 6. Out-of-Sample Evaluation

We evaluate on the held-out 20 % test set (19,159 orders, never seen during fitting).

**Threshold choice:**  
Default 0.50 is useless on an 8 % positive class — nearly everything gets predicted 'on-time'.  
We use the **Youden J threshold (~0.086)**, which maximises sensitivity + specificity.
This is appropriate for an operations setting where missing a late order (false negative)
is more costly than a false alarm.
"""))

cells.append(nbf.v4.new_code_cell("""\
y_prob = model_a.predict(X_test_c)

# Metrics
auc    = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
print(f"ROC-AUC:          {auc:.4f}  (random = 0.50)")
print(f"PR-AUC:           {pr_auc:.4f}  (random = {y_test.mean():.4f})")
print(f"McFadden Pseudo R²: {model_a.prsquared:.4f}")

# Threshold comparison (from saved CSV)
thresh_df = pd.read_csv('../outputs/model_a_threshold_comparison.csv')
print("\\n=== Threshold Comparison ===")
display(thresh_df[['Threshold_Name','Threshold','TN','FP','FN','TP',
                   'Precision','Recall (Sensitivity)','F1_Score']].round(3))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Confusion matrix at Youden threshold
THRESHOLD = 0.0855
y_pred = (y_prob >= THRESHOLD).astype(int)
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['On-Time', 'Late'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
tn, fp, fn, tp = cm.ravel()
prec = tp/(tp+fp); rec = tp/(tp+fn); f1 = 2*prec*rec/(prec+rec)
axes[0].set_title(
    f'Confusion Matrix (threshold = {THRESHOLD})\\n'
    f'Precision {prec:.1%}  Recall {rec:.1%}  F1 {f1:.3f}',
    fontsize=11)

# ROC curve
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr_arr, tpr_arr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
axes[1].plot([0,1],[0,1],'--', color='gray', label='Random')
axes[1].axvline(fp/(fp+tn), color='orange', linestyle=':', label=f'Youden threshold FPR')
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve — Model A (Late Delivery)', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/model_a_roc_cm.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/model_a_roc_cm.png")
"""))

# -------------------------------------------------------------------
# Section 7 — Robustness Checks
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 7. Robustness Checks

We test whether the core findings hold under alternate modelling choices.
The robustness checks were pre-run by `notebooks/run_robustness_checks.py`.

**Checks:**
1. **Alternate lateness definition** — `days_late ≥ 3` (vs > 0 baseline)
2. **Single-seller only** — drop orders with multiple sellers to remove aggregation noise
3. **Exclude Black Friday month** (Nov 2017) — removes demand shock outlier
4. **Delay-bucket model** — replace binary `is_late` with severity buckets (1-3, 4-7, 8-14, 15+ days)
"""))

cells.append(nbf.v4.new_code_cell("""\
rob_summary = pd.read_csv('../outputs/robustness_checks_summary.csv')
rob_buckets = pd.read_csv('../outputs/robustness_delay_buckets.csv')

print("=== Robustness Summary: Does the conclusion hold? ===")
display(rob_summary[['Model_Specification','Sample_Size','Target_Rate',
                      'Late_Odds_Ratio','Late_OR_CI95','OLS_Late_Penalty',
                      'Logit_Pseudo_R2']].round(4))

print("\\n=== Delay-Bucket Model: OR by Severity ===")
display(rob_buckets[['Delay_Bucket','Odds_Ratio','CI_Lower_95','CI_Upper_95','OLS_Drop_Stars']].round(3))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Plot delay buckets
fig, ax = plt.subplots(figsize=(8, 4))
labels  = rob_buckets['Delay_Bucket']
ors     = rob_buckets['Odds_Ratio']
cis_lo  = ors - rob_buckets['CI_Lower_95']
cis_hi  = rob_buckets['CI_Upper_95'] - ors
colors  = ['#f4a261','#e76f51','#c1121f','#9d0208']

bars = ax.barh(labels, ors, xerr=[cis_lo, cis_hi],
               color=colors, capsize=5, error_kw={'elinewidth':1.5})
ax.axvline(1, color='black', linestyle='--', linewidth=1, label='OR = 1')
for bar, val in zip(bars, ors):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}×', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Odds Ratio for Low Review (vs on-time)', fontsize=11)
ax.set_title('Delay Severity Drives Exponential Rating Damage\\n'
             '4-7 day delays → 15× higher odds of a 1-2 star review', fontsize=12)
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, rob_buckets['CI_Upper_95'].max() + 3)
plt.tight_layout()
plt.savefig('../outputs/model_a_delay_bucket_robustness.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/model_a_delay_bucket_robustness.png")
"""))

# -------------------------------------------------------------------
# Section 8 — Summary Table
# -------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 8. Model A Summary

| Metric | Value |
|:---|:---|
| **Training observations** | 76,636 |
| **Test observations** | 19,159 |
| **Positive rate (is_late)** | 8.1 % |
| **ROC-AUC (test)** | **0.7740** |
| **PR-AUC (test)** | 0.3145 |
| **McFadden Pseudo R²** | 0.1685 |
| **Max VIF** | 5.15 (cust_state_SP only) |
| **Convergence** | Yes (MLE, 200 iterations) |

### Top findings from Model A

| Factor | Odds Ratio | 95 % CI | Plain-English interpretation |
|:---|:---:|:---|:---|
| `seller_handling_days` | **2.22** | [2.17, 2.28] | Each extra std-dev of handling time **doubles** late odds |
| `log_distance` | 1.50 | [1.42, 1.58] | Longer routes add 50 % to late odds |
| `log_weight` | 1.06 | [1.01, 1.11] | Heavier parcels are marginally riskier |
| `estimated_days` | **0.27** | [0.25, 0.28] | Each extra buffer day **cuts** late odds by 73 % |
| `same_state` | 0.61 | [0.54, 0.69] | Intra-state orders are 39 % less likely to be late |
| `cust_state_PR` | 0.20 | [0.16, 0.24] | Paraná customers see the lowest late odds (carrier efficiency) |
| `cust_state_SP` | 0.21 | [0.18, 0.25] | São Paulo benefits from dense fulfillment hub proximity |
| `purchase_month` | 0.75 | [0.73, 0.77] | Later months (trend) associated with improving performance |

> **Association, not causation.** These coefficients show statistical association with lateness,  
> not confirmed causal pathways. See Limitations in `README.md`.
"""))

cells.append(nbf.v4.new_code_cell("""\
print("✓ Notebook 03_model_late.ipynb complete — all outputs saved to outputs/")
"""))

nb.cells = cells

# Write and execute
nb_path = os.path.join(os.path.dirname(__file__), '03_model_late.ipynb')
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
print(f"✓ Executed and saved: {nb_path}")
