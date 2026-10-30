import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shutil
import os

os.makedirs('outputs', exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Data & Prepare Features
# -----------------------------------------------------------------------------
print("Loading data from olist.duckdb (order_master)...")
con = duckdb.connect("olist.duckdb")
df = con.execute("SELECT * FROM order_master").df()

top_cust_states = df['customer_state'].value_counts().nlargest(10).index
df['cust_state_grouped'] = df['customer_state'].apply(lambda x: x if x in top_cust_states else 'other')

top_categories = df['category_en'].value_counts().nlargest(12).index
df['category_grouped'] = df['category_en'].apply(lambda x: x if x in top_categories else 'other')

df['log_weight'] = np.log1p(df['weight_capped'])
df['log_distance'] = np.log1p(df['distance_km'])
df['log_volume'] = np.log1p(df['volume_capped'])

continuous_cols = [
    'log_distance', 'log_weight', 'log_volume', 'n_items', 'n_sellers',
    'freight_ratio_capped', 'seller_handling_days', 'estimated_days', 'purchase_month'
]

# Use 'other' as natural baseline reference group
dummies_state = pd.get_dummies(df['cust_state_grouped'], prefix='cust_state', drop_first=False, dtype=float).drop(columns=['cust_state_other'])
dummies_cat = pd.get_dummies(df['category_grouped'], prefix='cat', drop_first=False, dtype=float).drop(columns=['cat_other'])

X_all = pd.concat([df[continuous_cols + ['same_state']], dummies_state, dummies_cat], axis=1)
y_all = df['is_late']

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# -----------------------------------------------------------------------------
# 2. VIF Analysis
# -----------------------------------------------------------------------------
print("\nCalculating Variance Inflation Factors (VIF)...")
vifs = []
for i, col in enumerate(X_train_const.columns):
    v = variance_inflation_factor(X_train_const.values, i)
    vifs.append((col, v))

vif_df = pd.DataFrame(vifs, columns=['Feature', 'VIF']).sort_values('VIF', ascending=False)
vif_df.to_csv('outputs/model_a_vif.csv', index=False)
print("Top 10 VIF values:")
print(vif_df.head(10).to_string(index=False))

# -----------------------------------------------------------------------------
# 3. Model Fit & Test Set Evaluation
# -----------------------------------------------------------------------------
print("\nFitting Logistic Regression...")
model = sm.Logit(y_train, X_train_const).fit(disp=False)
y_prob = model.predict(X_test_const)

test_auc = roc_auc_score(y_test, y_prob)
test_pr_auc = average_precision_score(y_test, y_prob)
base_rate = float(y_test.mean())

print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC:  {test_pr_auc:.4f}")
print(f"Base Rate:    {base_rate:.4f}")

# Threshold calculations
fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_j_idx = int(np.argmax(j_scores))
best_j_thresh = float(roc_thresh[best_j_idx])

prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
best_f1_idx = int(np.argmax(f1_scores))
best_f1_thresh = float(pr_thresh[best_f1_idx])

threshold_records = []
eval_dict = {
    'Default (0.50)': 0.50,
    'Base Rate (~0.081)': base_rate,
    "Youden's J (~0.086)": best_j_thresh,
    'Max F1 (~0.157)': best_f1_thresh
}

for name, th in eval_dict.items():
    pred = (y_prob >= th).astype(int)
    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f = f1_score(y_test, pred, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    threshold_records.append({
        'Threshold_Name': name,
        'Threshold': th,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'Accuracy': acc,
        'Precision': p,
        'Recall (Sensitivity)': r,
        'Specificity': spec,
        'F1_Score': f
    })

thresh_df = pd.DataFrame(threshold_records)
thresh_df.to_csv('outputs/model_a_threshold_comparison.csv', index=False)
print("\n--- Threshold Comparison Table ---")
print(thresh_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 4. Generate Comprehensive Diagnostic Visualizations
# -----------------------------------------------------------------------------
print("\nGenerating diagnostic plots...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
colors = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728', 'purple': '#9467bd', 'dark': '#2c3e50'}

# Panel 1: ROC Curve
ax1 = axes[0, 0]
ax1.plot(fpr, tpr, color=colors['blue'], lw=2.5, label=f'Model A ROC (AUC = {test_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
ax1.scatter(fpr[best_j_idx], tpr[best_j_idx], color=colors['red'], s=100, zorder=5,
            label=f"Youden's J Point (tau = {best_j_thresh:.3f})\nTPR = {tpr[best_j_idx]:.1%}, FPR = {fpr[best_j_idx]:.1%}")
ax1.set_title('A. Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', color=colors['dark'])
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax1.set_ylabel('True Positive Rate (Recall)', fontsize=10)
ax1.legend(loc='lower right', frameon=True, fontsize=9)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.02])

# Panel 2: Precision-Recall Curve
ax2 = axes[0, 1]
ax2.plot(rec, prec, color=colors['purple'], lw=2.5, label=f'Model A PR (AUC = {test_pr_auc:.4f})')
ax2.axhline(base_rate, color='gray', linestyle='--', lw=1.5, label=f'Base Rate Baseline = {base_rate:.1%}')
ax2.scatter(rec[best_f1_idx], prec[best_f1_idx], color=colors['green'], s=100, zorder=5,
            label=f"Max F1 Point (tau = {best_f1_thresh:.3f})\nF1 = {f1_scores[best_f1_idx]:.3f}, Prec = {prec[best_f1_idx]:.1%}, Rec = {rec[best_f1_idx]:.1%}")
ax2.set_title('B. Precision-Recall Curve (Imbalanced Failure Metric)', fontsize=12, fontweight='bold', color=colors['dark'])
ax2.set_xlabel('Recall (Sensitivity)', fontsize=10)
ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=10)
ax2.legend(loc='upper right', frameon=True, fontsize=9)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 0.75])

# Panel 3: Confusion Matrix at Sensible Operational Threshold
pred_sensible = (y_prob >= best_j_thresh).astype(int)
cm_sensible = confusion_matrix(y_test, pred_sensible)

ax3 = axes[1, 0]
sns.heatmap(cm_sensible, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax3,
            annot_kws={'size': 13, 'weight': 'bold'})
ax3.set_title(f'C. Confusion Matrix at Sensible Alert Threshold (tau = {best_j_thresh:.3f})\nCatches 67.2% of Late Deliveries ({cm_sensible[1, 1]:,} / {cm_sensible[1].sum():,})',
              fontsize=12, fontweight='bold', color=colors['dark'])
ax3.set_xlabel('Predicted Label (0 = On-Time, 1 = Late)', fontsize=10)
ax3.set_ylabel('Actual True Label (0 = On-Time, 1 = Late)', fontsize=10)
ax3.set_xticklabels(['Pred On-Time (0)', 'Pred Late (1)'])
ax3.set_yticklabels(['True On-Time (0)', 'True Late (1)'])

tn, fp, fn, tp = cm_sensible.ravel()
ax3.text(0.5, -0.16, f'True Positives: {tp:,} | False Positives: {fp:,}\nFalse Negatives: {fn:,} | True Negatives: {tn:,}',
         transform=ax3.transAxes, ha='center', fontsize=9, style='italic', color='#444')

# Panel 4: Metric Tradeoff vs Threshold
eval_thresholds = np.linspace(0.02, 0.60, 60)
acc_list, prec_list, rec_list, f1_list = [], [], [], []

for t in eval_thresholds:
    p_bin = (y_prob >= t).astype(int)
    acc_list.append(accuracy_score(y_test, p_bin))
    prec_list.append(precision_score(y_test, p_bin, zero_division=0))
    rec_list.append(recall_score(y_test, p_bin, zero_division=0))
    f1_list.append(f1_score(y_test, p_bin, zero_division=0))

ax4 = axes[1, 1]
ax4.plot(eval_thresholds, acc_list, label='Accuracy (Deceptively High)', color='gray', linestyle='--', lw=2)
ax4.plot(eval_thresholds, rec_list, label='Recall / Sensitivity (Catches Late)', color=colors['blue'], lw=2.5)
ax4.plot(eval_thresholds, prec_list, label='Precision', color=colors['orange'], lw=2)
ax4.plot(eval_thresholds, f1_list, label='F1-Score', color=colors['green'], lw=2.5)
ax4.axvline(0.50, color=colors['red'], linestyle=':', lw=1.5, label='Default tau = 0.50 (Recall collapses to 8.8%)')
ax4.axvline(best_j_thresh, color=colors['dark'], linestyle='-', lw=1.5, label=f'Sensible tau = {best_j_thresh:.3f} (Balanced Alert)')
ax4.set_title('D. Metric Tradeoff Across Classification Thresholds', fontsize=12, fontweight='bold', color=colors['dark'])
ax4.set_xlabel('Decision Threshold (tau)', fontsize=10)
ax4.set_ylabel('Metric Value (0.0 to 1.0)', fontsize=10)
ax4.legend(loc='center right', frameon=True, fontsize=8.5)
ax4.set_xlim([0.02, 0.60])
ax4.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('outputs/model_a_diagnostics.png', dpi=300)
plt.close()

# Copy to brain artifact directory
artifact_dir = r'C:\Users\USER\.gemini\antigravity\brain\9c3f3cb4-a5ab-4607-a780-d83142b229e5'
if os.path.exists(artifact_dir):
    shutil.copy('outputs/model_a_diagnostics.png', os.path.join(artifact_dir, 'model_a_diagnostics.png'))

print("Completed successfully! Saved outputs/model_a_diagnostics.png and outputs/model_a_threshold_comparison.csv.")
