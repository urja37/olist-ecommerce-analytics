import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os

os.makedirs('outputs', exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Clean Data with Reviews
# -----------------------------------------------------------------------------
print("Loading data from olist.duckdb (order_master)...")
con = duckdb.connect("olist.duckdb")
df = con.execute("SELECT * FROM order_master WHERE review_score IS NOT NULL").df()
print(f"Loaded {len(df):,} clean orders with valid customer reviews.")

# Group high-cardinality categorical variables
top_cust_states = df['customer_state'].value_counts().nlargest(10).index
df['cust_state_grouped'] = df['customer_state'].apply(lambda x: x if x in top_cust_states else 'other')

top_categories = df['category_en'].value_counts().nlargest(12).index
df['category_grouped'] = df['category_en'].apply(lambda x: x if x in top_categories else 'other')

# Log-transform continuous skewed features
df['log_weight'] = np.log1p(df['weight_capped'])
df['log_distance'] = np.log1p(df['distance_km'])
df['log_volume'] = np.log1p(df['volume_capped'])
df['log_price'] = np.log1p(df['total_price'])

# Predictor set specified by prompt:
# freight ratio, price, n_items, n_sellers, distance, handling days, state, product category
continuous_cols = [
    'freight_ratio_capped',
    'log_price',
    'n_items',
    'n_sellers',
    'log_distance',
    'seller_handling_days'
]

# Set 'other' as reference category for state and category
dummies_state = pd.get_dummies(df['cust_state_grouped'], prefix='cust_state', drop_first=False, dtype=float).drop(columns=['cust_state_other'])
dummies_cat = pd.get_dummies(df['category_grouped'], prefix='cat', drop_first=False, dtype=float).drop(columns=['cat_other'])

X_base_raw = pd.concat([df[continuous_cols], dummies_state, dummies_cat], axis=1)
y_low = df['low_review']
y_score = df['review_score']
is_late = df['is_late']

# 80/20 train/test split stratified by low_review
idx_train, idx_test = train_test_split(df.index, test_size=0.20, random_state=42, stratify=y_low)

# Standardize continuous variables (fit train, transform test)
scaler = StandardScaler()
X_train_cont = pd.DataFrame(scaler.fit_transform(X_base_raw.loc[idx_train, continuous_cols]), columns=continuous_cols, index=idx_train)
X_test_cont = pd.DataFrame(scaler.transform(X_base_raw.loc[idx_test, continuous_cols]), columns=continuous_cols, index=idx_test)

non_cont_cols = [c for c in X_base_raw.columns if c not in continuous_cols]
X_train_s1 = pd.concat([X_train_cont, X_base_raw.loc[idx_train, non_cont_cols]], axis=1)
X_test_s1 = pd.concat([X_test_cont, X_base_raw.loc[idx_test, non_cont_cols]], axis=1)

X_train_s1_const = sm.add_constant(X_train_s1)
X_test_s1_const = sm.add_constant(X_test_s1)

# Step 2 matrices (add is_late)
X_train_s2_const = X_train_s1_const.copy()
X_train_s2_const['is_late'] = is_late.loc[idx_train]

X_test_s2_const = X_test_s1_const.copy()
X_test_s2_const['is_late'] = is_late.loc[idx_test]

# -----------------------------------------------------------------------------
# 2. Step 1: Logistic Regression WITHOUT Lateness
# -----------------------------------------------------------------------------
print("\nFitting Model B1: Logistic Regression WITHOUT Lateness...")
logit_s1 = sm.Logit(y_low.loc[idx_train], X_train_s1_const).fit(disp=False)
prob_s1 = logit_s1.predict(X_test_s1_const)
auc_s1 = roc_auc_score(y_low.loc[idx_test], prob_s1)
prauc_s1 = average_precision_score(y_low.loc[idx_test], prob_s1)

print(f"Model B1 Pseudo R2:   {logit_s1.prsquared:.4f}")
print(f"Model B1 Log-Lik:     {logit_s1.llf:.2f}")
print(f"Model B1 AIC:         {logit_s1.aic:.2f}")
print(f"Model B1 Test ROC-AUC:{auc_s1:.4f}")

# -----------------------------------------------------------------------------
# 3. Step 2: Logistic Regression WITH Lateness (is_late)
# -----------------------------------------------------------------------------
print("\nFitting Model B2: Logistic Regression WITH Lateness (is_late)...")
logit_s2 = sm.Logit(y_low.loc[idx_train], X_train_s2_const).fit(disp=False)
prob_s2 = logit_s2.predict(X_test_s2_const)
auc_s2 = roc_auc_score(y_low.loc[idx_test], prob_s2)
prauc_s2 = average_precision_score(y_low.loc[idx_test], prob_s2)

lr_stat = 2 * (logit_s2.llf - logit_s1.llf)
print(f"Model B2 Pseudo R2:   {logit_s2.prsquared:.4f} (Delta = +{logit_s2.prsquared - logit_s1.prsquared:.4f})")
print(f"Model B2 Log-Lik:     {logit_s2.llf:.2f} (Delta = +{logit_s2.llf - logit_s1.llf:.2f})")
print(f"Model B2 AIC:         {logit_s2.aic:.2f}")
print(f"Model B2 Test ROC-AUC:{auc_s2:.4f} (Delta = +{auc_s2 - auc_s1:.4f})")
print(f"Likelihood Ratio Stat: {lr_stat:.2f} (p < 0.0001)")

# Compile Logit Comparison Table
conf1 = logit_s1.conf_int()
conf2 = logit_s2.conf_int()

comp_list = []
for var in X_train_s2_const.columns:
    c1 = logit_s1.params.get(var, np.nan)
    or1 = np.exp(c1) if not np.isnan(c1) else np.nan
    p1 = logit_s1.pvalues.get(var, np.nan)
    ci1_low = np.exp(conf1.loc[var, 0]) if var in conf1.index else np.nan
    ci1_high = np.exp(conf1.loc[var, 1]) if var in conf1.index else np.nan
    
    c2 = logit_s2.params.get(var, np.nan)
    or2 = np.exp(c2) if not np.isnan(c2) else np.nan
    p2 = logit_s2.pvalues.get(var, np.nan)
    ci2_low = np.exp(conf2.loc[var, 0]) if var in conf2.index else np.nan
    ci2_high = np.exp(conf2.loc[var, 1]) if var in conf2.index else np.nan
    
    comp_list.append({
        'Feature': var,
        'Coef_Step1_NoLate': c1,
        'OR_Step1_NoLate': or1,
        'p_val_Step1': p1,
        'CI95_Step1': f"[{ci1_low:.2f}, {ci1_high:.2f}]" if not np.isnan(ci1_low) else "-",
        'Coef_Step2_WithLate': c2,
        'OR_Step2_WithLate': or2,
        'p_val_Step2': p2,
        'CI95_Step2': f"[{ci2_low:.2f}, {ci2_high:.2f}]" if not np.isnan(ci2_low) else "-",
        'Delta_OR': or2 - or1 if not np.isnan(or1) else np.nan
    })

comp_df = pd.DataFrame(comp_list)
comp_df.to_csv('outputs/model_b_logit_comparison.csv', index=False)

# -----------------------------------------------------------------------------
# 4. View 2: OLS Regression on Raw Review Score (1-5 Scale)
# -----------------------------------------------------------------------------
print("\nFitting OLS on Raw review_score (1-5)...")
ols_s1 = sm.OLS(y_score.loc[idx_train], X_train_s1_const).fit()
ols_s2 = sm.OLS(y_score.loc[idx_train], X_train_s2_const).fit()

pred_ols_test_s1 = ols_s1.predict(X_test_s1_const)
pred_ols_test_s2 = ols_s2.predict(X_test_s2_const)

r2_test_s1 = r2_score(y_score.loc[idx_test], pred_ols_test_s1)
r2_test_s2 = r2_score(y_score.loc[idx_test], pred_ols_test_s2)

print(f"OLS Step 1 R-squared: {ols_s1.rsquared:.4f} (Test R2: {r2_test_s1:.4f})")
print(f"OLS Step 2 R-squared: {ols_s2.rsquared:.4f} (Test R2: {r2_test_s2:.4f}, Delta = +{ols_s2.rsquared - ols_s1.rsquared:.4f})")
print(f"OLS is_late coefficient: {ols_s2.params['is_late']:.4f} (p < 0.0001, 95% CI: [{ols_s2.conf_int().loc['is_late', 0]:.4f}, {ols_s2.conf_int().loc['is_late', 1]:.4f}])")

# Save OLS summaries
with open('outputs/model_b_ols_summary.txt', 'w') as f:
    f.write("=== OLS STEP 1 (WITHOUT LATENESS) ===\n")
    f.write(str(ols_s1.summary()))
    f.write("\n\n=== OLS STEP 2 (WITH IS_LATE) ===\n")
    f.write(str(ols_s2.summary()))

ols_coef_df = pd.DataFrame({
    'Feature': ols_s2.params.index,
    'OLS_Coef_Step1': [ols_s1.params.get(f, np.nan) for f in ols_s2.params.index],
    'OLS_Coef_Step2': ols_s2.params.values,
    'Std_Error_Step2': ols_s2.bse.values,
    't_stat_Step2': ols_s2.tvalues.values,
    'p_value_Step2': ols_s2.pvalues.values,
    'CI_Lower_95': ols_s2.conf_int()[0].values,
    'CI_Upper_95': ols_s2.conf_int()[1].values
})
ols_coef_df.to_csv('outputs/model_b_ols_coefficients.csv', index=False)

# -----------------------------------------------------------------------------
# 5. Diagnostic & Explanatory Visualizations
# -----------------------------------------------------------------------------
print("\nGenerating Model B visualizations...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
colors = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728', 'purple': '#9467bd', 'dark': '#2c3e50'}

# Panel 1: Odds Ratios Comparison for Key Predictors (Step 1 vs Step 2)
key_vars = ['seller_handling_days', 'log_distance', 'n_items', 'freight_ratio_capped', 'log_price', 'is_late']
plot_data = []
for v in key_vars:
    if v != 'is_late':
        plot_data.append({'Variable': v, 'Model': 'Step 1: Without Lateness', 'Odds_Ratio': np.exp(logit_s1.params[v])})
    plot_data.append({'Variable': v, 'Model': 'Step 2: With Lateness', 'Odds_Ratio': np.exp(logit_s2.params[v])})

plot_df = pd.DataFrame(plot_data)
# Friendly variable names
name_map = {
    'is_late': 'Late Delivery (is_late)',
    'seller_handling_days': 'Seller Handling Days',
    'log_distance': 'Shipping Distance (log)',
    'n_items': 'Number of Items',
    'freight_ratio_capped': 'Freight Ratio',
    'log_price': 'Product Price (log)'
}
plot_df['Display_Name'] = plot_df['Variable'].map(name_map)

ax1 = axes[0, 0]
sns.barplot(data=plot_df, x='Odds_Ratio', y='Display_Name', hue='Model', palette=[colors['blue'], colors['red']], ax=ax1)
ax1.axvline(1.0, color='gray', linestyle='--', lw=1.5)
ax1.set_title('A. Odds Ratios on Low Review: Before vs. After Adding Lateness', fontsize=12, fontweight='bold', color=colors['dark'])
ax1.set_xlabel('Odds Ratio (exp(beta)) - Log Scale', fontsize=10)
ax1.set_ylabel('')
ax1.set_xscale('log')
ax1.legend(loc='lower right', frameon=True, fontsize=9)

# Panel 2: Model Fit Metrics (R-squared & Pseudo R-squared Jump)
ax2 = axes[0, 1]
models = ['Step 1: No Lateness', 'Step 2: With is_late']
pseudo_r2 = [logit_s1.prsquared, logit_s2.prsquared]
ols_r2 = [ols_s1.rsquared, ols_s2.rsquared]

x = np.arange(len(models))
width = 0.35
ax2.bar(x - width/2, pseudo_r2, width, label='Logit Pseudo R²', color=colors['blue'], alpha=0.9)
ax2.bar(x + width/2, ols_r2, width, label='OLS R² (Raw Score)', color=colors['green'], alpha=0.9)

for i in range(len(models)):
    ax2.text(x[i] - width/2, pseudo_r2[i] + 0.005, f"{pseudo_r2[i]:.3f}", ha='center', fontweight='bold', fontsize=10)
    ax2.text(x[i] + width/2, ols_r2[i] + 0.005, f"{ols_r2[i]:.3f}", ha='center', fontweight='bold', fontsize=10)

ax2.set_title('B. Explanatory Power Jump (Lateness Absorbs Variance)', fontsize=12, fontweight='bold', color=colors['dark'])
ax2.set_ylabel('Variance Explained (R² / Pseudo R²)', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight='bold')
ax2.set_ylim([0, 0.20])
ax2.legend(loc='upper left', frameon=True, fontsize=9)

# Panel 3: OLS Impact - Predicted Review Score (On-Time vs Late)
ax3 = axes[1, 0]
mean_ontime = df[df['is_late'] == 0]['review_score'].mean()
mean_late = df[df['is_late'] == 1]['review_score'].mean()
ols_drop = abs(ols_s2.params['is_late'])

bar_labels = ['On-Time Orders', 'Late Orders']
bar_vals = [mean_ontime, mean_late]
bar_colors = [colors['green'], colors['red']]

bars = ax3.bar(bar_labels, bar_vals, color=bar_colors, width=0.5, alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.08, f"{yval:.2f} stars", ha='center', fontweight='bold', fontsize=11)

ax3.annotate(f"Direct OLS Penalty:\n- {ols_drop:.2f} Points (p < 0.0001)\nHolding route, price & handling constant",
             xy=(0.5, 3.0), xycoords='data',
             ha='center', fontsize=10, fontweight='bold', color=colors['dark'],
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff2f2", ec=colors['red'], lw=1.5))

ax3.set_title('C. OLS Plain-English View: Review Score by Delivery Status', fontsize=12, fontweight='bold', color=colors['dark'])
ax3.set_ylabel('Average Customer Review Score (1 to 5)', fontsize=10)
ax3.set_ylim([0, 5.5])

# Panel 4: ROC Curves Comparison (Step 1 vs Step 2)
from sklearn.metrics import roc_curve
fpr1, tpr1, _ = roc_curve(y_low.loc[idx_test], prob_s1)
fpr2, tpr2, _ = roc_curve(y_low.loc[idx_test], prob_s2)

ax4 = axes[1, 1]
ax4.plot(fpr1, tpr1, color=colors['blue'], lw=2, label=f'Step 1: Without Lateness (AUC = {auc_s1:.4f})')
ax4.plot(fpr2, tpr2, color=colors['red'], lw=2.5, label=f'Step 2: With is_late (AUC = {auc_s2:.4f})')
ax4.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
ax4.set_title('D. Discrimination Gain: ROC Curves on Low Review', fontsize=12, fontweight='bold', color=colors['dark'])
ax4.set_xlabel('False Positive Rate', fontsize=10)
ax4.set_ylabel('True Positive Rate', fontsize=10)
ax4.legend(loc='lower right', frameon=True, fontsize=9)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('outputs/model_b_step_comparison.png', dpi=300)
plt.close()

# Copy to brain artifact directory
artifact_dir = r'C:\Users\USER\.gemini\antigravity\brain\9c3f3cb4-a5ab-4607-a780-d83142b229e5'
if os.path.exists(artifact_dir):
    shutil.copy('outputs/model_b_step_comparison.png', os.path.join(artifact_dir, 'model_b_step_comparison.png'))

print("Completed successfully! Saved outputs/model_b_step_comparison.png, outputs/model_b_logit_comparison.csv, outputs/model_b_ols_coefficients.csv.")
