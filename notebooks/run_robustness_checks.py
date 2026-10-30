import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os

os.makedirs('outputs', exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------
print("Loading data from olist.duckdb (order_master)...")
con = duckdb.connect("olist.duckdb")
df = con.execute("SELECT * FROM order_master WHERE review_score IS NOT NULL").df()

# Group high-cardinality categories
top_cust_states = df['customer_state'].value_counts().nlargest(10).index
df['cust_state_grouped'] = df['customer_state'].apply(lambda x: x if x in top_cust_states else 'other')

top_categories = df['category_en'].value_counts().nlargest(12).index
df['category_grouped'] = df['category_en'].apply(lambda x: x if x in top_categories else 'other')

# Log-transform continuous skewed features
df['log_weight'] = np.log1p(df['weight_capped'])
df['log_distance'] = np.log1p(df['distance_km'])
df['log_volume'] = np.log1p(df['volume_capped'])
df['log_price'] = np.log1p(df['total_price'])

# Add delay metrics
df['days_late_pos'] = df['days_late'].apply(lambda x: max(0.0, x))
df['purchase_yr_mo'] = pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M').astype(str)

# Delay buckets
def get_delay_bucket(row):
    if row['is_late'] == 0 or row['days_late'] <= 0:
        return 'on_time'
    elif row['days_late'] <= 3:
        return 'late_1_3'
    elif row['days_late'] <= 7:
        return 'late_4_7'
    elif row['days_late'] <= 14:
        return 'late_8_14'
    else:
        return 'late_15_plus'

df['delay_bucket'] = df.apply(get_delay_bucket, axis=1)

# Continuous columns to standardize
continuous_cols = [
    'freight_ratio_capped',
    'log_price',
    'n_items',
    'n_sellers',
    'log_distance',
    'seller_handling_days'
]

dummies_state = pd.get_dummies(df['cust_state_grouped'], prefix='cust_state', drop_first=False, dtype=float).drop(columns=['cust_state_other'])
dummies_cat = pd.get_dummies(df['category_grouped'], prefix='cat', drop_first=False, dtype=float).drop(columns=['cat_other'])

X_base = pd.concat([df[continuous_cols], dummies_state, dummies_cat], axis=1)

# Helper function to run Logit & OLS
def fit_models(X_df, y_binary, y_cont, late_col, sample_name=""):
    # Standardize continuous
    scaler = StandardScaler()
    X_scaled = X_df.copy()
    cont_in_df = [c for c in continuous_cols if c in X_df.columns]
    X_scaled[cont_in_df] = scaler.fit_transform(X_scaled[cont_in_df])
    
    # Add late indicator / predictor
    if isinstance(late_col, str):
        X_scaled[late_col] = df.loc[X_df.index, late_col]
    elif isinstance(late_col, pd.DataFrame):
        X_scaled = pd.concat([X_scaled, late_col], axis=1)
        
    X_const = sm.add_constant(X_scaled)
    
    # Fit Logit
    logit_m = sm.Logit(y_binary, X_const).fit(disp=False)
    
    # Fit OLS
    ols_m = sm.OLS(y_cont, X_const).fit()
    
    return logit_m, ols_m, X_scaled

# -----------------------------------------------------------------------------
# Baseline Model: low_review <= 2 with is_late
# -----------------------------------------------------------------------------
print("\n--- Fitting Baseline Model (low_review <= 2) ---")
logit_base, ols_base, _ = fit_models(X_base, df['low_review'], df['review_score'], 'is_late', "Baseline")

# -----------------------------------------------------------------------------
# Check 1: Redefine low review as <= 3 (Broad Dissatisfaction)
# -----------------------------------------------------------------------------
print("\n--- Check 1: Redefining low_review as review_score <= 3 ---")
df['low_review_3'] = (df['review_score'] <= 3).astype(int)
logit_c1, ols_c1, _ = fit_models(X_base, df['low_review_3'], df['review_score'], 'is_late', "Check 1: review <= 3")

# -----------------------------------------------------------------------------
# Check 2: Use days_late (Delay Buckets) - Is 10 days worse than 1 day?
# -----------------------------------------------------------------------------
print("\n--- Check 2: Delay Severity (Categorical Delay Buckets) ---")
delay_dummies = pd.get_dummies(df['delay_bucket'], prefix='delay', drop_first=False, dtype=float)
# Reference is on_time
delay_dummies = delay_dummies.drop(columns=['delay_on_time'])

logit_c2, ols_c2, _ = fit_models(X_base, df['low_review'], df['review_score'], delay_dummies, "Check 2: Delay Buckets")

# Continuous days late for positive delays
df['log_days_late_pos'] = np.log1p(df['days_late_pos'])
logit_c2_cont, ols_c2_cont, _ = fit_models(X_base, df['low_review'], df['review_score'], 'days_late_pos', "Check 2: Cont Days Late")

# -----------------------------------------------------------------------------
# Check 3: Drop Multi-Seller Orders (n_sellers == 1 only)
# -----------------------------------------------------------------------------
print("\n--- Check 3: Single-Seller Orders Only ---")
single_idx = df[df['n_sellers'] == 1].index
X_single = X_base.loc[single_idx].drop(columns=['n_sellers'])
logit_c3, ols_c3, _ = fit_models(X_single, df.loc[single_idx, 'low_review'], df.loc[single_idx, 'review_score'], 'is_late', "Check 3: Single Seller")

# -----------------------------------------------------------------------------
# Check 4: Exclude Black Friday Month (Nov 2017)
# -----------------------------------------------------------------------------
print("\n--- Check 4: Exclude Black Friday (Nov 2017) ---")
nobf_idx = df[df['purchase_yr_mo'] != '2017-11'].index
X_nobf = X_base.loc[nobf_idx]
logit_c4, ols_c4, _ = fit_models(X_nobf, df.loc[nobf_idx, 'low_review'], df.loc[nobf_idx, 'review_score'], 'is_late', "Check 4: Exclude Nov 2017")

# -----------------------------------------------------------------------------
# Compile Robustness Summary Table
# -----------------------------------------------------------------------------
summary_rows = [
    {
        'Model_Specification': 'Baseline (Full Sample, low_review <= 2)',
        'Sample_Size': len(df),
        'Target_Rate': f"{df['low_review'].mean():.1%}",
        'Late_Odds_Ratio': f"{np.exp(logit_base.params['is_late']):.2f}",
        'Late_OR_CI95': f"[{np.exp(logit_base.conf_int().loc['is_late', 0]):.2f}, {np.exp(logit_base.conf_int().loc['is_late', 1]):.2f}]",
        'OLS_Late_Penalty': f"{ols_base.params['is_late']:.2f} stars",
        'Logit_Pseudo_R2': f"{logit_base.prsquared:.4f}",
        'OLS_R2': f"{ols_base.rsquared:.4f}"
    },
    {
        'Model_Specification': 'Check 1: Low Review Defined as <= 3',
        'Sample_Size': len(df),
        'Target_Rate': f"{df['low_review_3'].mean():.1%}",
        'Late_Odds_Ratio': f"{np.exp(logit_c1.params['is_late']):.2f}",
        'Late_OR_CI95': f"[{np.exp(logit_c1.conf_int().loc['is_late', 0]):.2f}, {np.exp(logit_c1.conf_int().loc['is_late', 1]):.2f}]",
        'OLS_Late_Penalty': f"{ols_c1.params['is_late']:.2f} stars",
        'Logit_Pseudo_R2': f"{logit_c1.prsquared:.4f}",
        'OLS_R2': f"{ols_c1.rsquared:.4f}"
    },
    {
        'Model_Specification': 'Check 3: Single-Seller Orders Only (Drop Multi-Seller)',
        'Sample_Size': len(single_idx),
        'Target_Rate': f"{df.loc[single_idx, 'low_review'].mean():.1%}",
        'Late_Odds_Ratio': f"{np.exp(logit_c3.params['is_late']):.2f}",
        'Late_OR_CI95': f"[{np.exp(logit_c3.conf_int().loc['is_late', 0]):.2f}, {np.exp(logit_c3.conf_int().loc['is_late', 1]):.2f}]",
        'OLS_Late_Penalty': f"{ols_c3.params['is_late']:.2f} stars",
        'Logit_Pseudo_R2': f"{logit_c3.prsquared:.4f}",
        'OLS_R2': f"{ols_c3.rsquared:.4f}"
    },
    {
        'Model_Specification': 'Check 4: Exclude Black Friday Month (Nov 2017)',
        'Sample_Size': len(nobf_idx),
        'Target_Rate': f"{df.loc[nobf_idx, 'low_review'].mean():.1%}",
        'Late_Odds_Ratio': f"{np.exp(logit_c4.params['is_late']):.2f}",
        'Late_OR_CI95': f"[{np.exp(logit_c4.conf_int().loc['is_late', 0]):.2f}, {np.exp(logit_c4.conf_int().loc['is_late', 1]):.2f}]",
        'OLS_Late_Penalty': f"{ols_c4.params['is_late']:.2f} stars",
        'Logit_Pseudo_R2': f"{logit_c4.prsquared:.4f}",
        'OLS_R2': f"{ols_c4.rsquared:.4f}"
    }
]

rob_df = pd.DataFrame(summary_rows)
rob_df.to_csv('outputs/robustness_checks_summary.csv', index=False)
print("\n=== ROBUSTNESS CHECKS SUMMARY TABLE ===")
print(rob_df.to_string(index=False))

# Check 2: Detailed Delay Buckets Table
delay_table = pd.DataFrame({
    'Delay_Bucket': ['1 to 3 days late', '4 to 7 days late', '8 to 14 days late', '15+ days late'],
    'Variable': ['delay_late_1_3', 'delay_late_4_7', 'delay_late_8_14', 'delay_late_15_plus'],
    'Logit_Coef': [logit_c2.params[v] for v in ['delay_late_1_3', 'delay_late_4_7', 'delay_late_8_14', 'delay_late_15_plus']],
    'Odds_Ratio': [np.exp(logit_c2.params[v]) for v in ['delay_late_1_3', 'delay_late_4_7', 'delay_late_8_14', 'delay_late_15_plus']],
    'CI_Lower_95': [np.exp(logit_c2.conf_int().loc[v, 0]) for v in ['delay_late_1_3', 'delay_late_4_7', 'delay_late_8_14', 'delay_late_15_plus']],
    'CI_Upper_95': [np.exp(logit_c2.conf_int().loc[v, 1]) for v in ['delay_late_1_3', 'delay_late_4_7', 'delay_late_8_14', 'delay_late_15_plus']],
    'OLS_Drop_Stars': [ols_c2.params[v] for v in ['delay_late_1_3', 'delay_late_4_7', 'delay_late_8_14', 'delay_late_15_plus']]
})
delay_table.to_csv('outputs/robustness_delay_buckets.csv', index=False)
print("\n=== CHECK 2: DELAY SEVERITY (IS 10 DAYS WORSE THAN 1 DAY?) ===")
print(delay_table.to_string(index=False))

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
print("\nGenerating Robustness Visualizations...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
colors = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728', 'purple': '#9467bd', 'dark': '#2c3e50'}

# Panel 1: Check 1 & Specification Stability (Odds Ratio Forest Plot)
ax1 = axes[0, 0]
models_plot = ['Baseline (<=2)', 'Check 1 (<=3)', 'Check 3 (Single Seller)', 'Check 4 (No Black Friday)']
ors = [float(r['Late_Odds_Ratio']) for r in summary_rows]
ci_lows = [float(r['Late_OR_CI95'].split(',')[0].replace('[', '')) for r in summary_rows]
ci_highs = [float(r['Late_OR_CI95'].split(',')[1].replace(']', '')) for r in summary_rows]

y_pos = np.arange(len(models_plot))
ax1.errorbar(ors, y_pos, xerr=[np.array(ors) - np.array(ci_lows), np.array(ci_highs) - np.array(ors)],
             fmt='o', color=colors['red'], ecolor=colors['red'], elinewidth=2.5, capsize=6, markersize=8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models_plot, fontweight='bold')
ax1.set_xlabel('Odds Ratio of Lateness (exp(beta)) with 95% CI', fontsize=10)
ax1.set_title('A. Robustness of Late Delivery Penalty Across Specifications', fontsize=12, fontweight='bold', color=colors['dark'])
for i, v in enumerate(ors):
    ax1.text(v + 0.25, i, f"OR = {v:.2f}", va='center', fontweight='bold', fontsize=10)
ax1.set_xlim([4, 13])

# Panel 2: Check 2 - Odds Ratio by Delay Duration (Is 10 days worse than 1?)
ax2 = axes[0, 1]
delay_labels = ['On-Time\n(Reference)', '1 to 3\nDays Late', '4 to 7\nDays Late', '8 to 14\nDays Late', '15+\nDays Late']
delay_ors = [1.0] + delay_table['Odds_Ratio'].tolist()
delay_err_low = [0.0] + (delay_table['Odds_Ratio'] - delay_table['CI_Lower_95']).tolist()
delay_err_high = [0.0] + (delay_table['CI_Upper_95'] - delay_table['Odds_Ratio']).tolist()

bars2 = ax2.bar(delay_labels, delay_ors, color=[colors['green']] + [colors['orange']] + [colors['red']]*3, alpha=0.85, width=0.55)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.6, f"{h:.1f}x", ha='center', fontweight='bold', fontsize=10)

ax2.set_title('B. Delay Severity: Odds of Bad Review vs. Days Late', fontsize=12, fontweight='bold', color=colors['dark'])
ax2.set_ylabel('Odds Ratio vs. On-Time Baseline', fontsize=10)
ax2.set_ylim([0, 32])

# Panel 3: Check 2 - OLS Review Score Penalty by Delay Duration
ax3 = axes[1, 0]
raw_scores_bucket = [
    df[df['delay_bucket'] == 'on_time']['review_score'].mean(),
    df[df['delay_bucket'] == 'late_1_3']['review_score'].mean(),
    df[df['delay_bucket'] == 'late_4_7']['review_score'].mean(),
    df[df['delay_bucket'] == 'late_8_14']['review_score'].mean(),
    df[df['delay_bucket'] == 'late_15_plus']['review_score'].mean()
]

bars3 = ax3.plot(delay_labels, raw_scores_bucket, marker='s', markersize=9, lw=2.5, color=colors['blue'], label='Observed Mean Score')
for i, txt in enumerate(raw_scores_bucket):
    ax3.annotate(f"{txt:.2f} stars", (delay_labels[i], txt), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

ax3.axhline(4.29, color='gray', linestyle='--', label='On-Time Baseline (4.29 stars)')
ax3.set_title('C. Customer Rating Plunge: Review Score by Delay Duration', fontsize=12, fontweight='bold', color=colors['dark'])
ax3.set_ylabel('Average Review Score (1 to 5 Stars)', fontsize=10)
ax3.set_ylim([1.0, 5.0])
ax3.legend(loc='lower left', frameon=True, fontsize=9)

# Panel 4: Check 4 - Monthly Late Rate vs Black Friday Anomaly
ax4 = axes[1, 1]
monthly_data = df.groupby('purchase_yr_mo').agg(
    total_orders=('order_id', 'count'),
    late_rate=('is_late', 'mean'),
    avg_score=('review_score', 'mean')
).reset_index()
monthly_data = monthly_data[monthly_data['total_orders'] >= 100]

months = monthly_data['purchase_yr_mo']
x_m = np.arange(len(months))
ax4.plot(x_m, monthly_data['late_rate'] * 100, marker='o', color=colors['red'], lw=2, label='Late Delivery Rate (%)')
ax4.set_xticks(x_m[::2])
ax4.set_xticklabels(months[::2], rotation=45, ha='right', fontsize=9)

# Highlight Nov 2017
nov_idx = months[months == '2017-11'].index[0]
ax4.scatter([nov_idx], [monthly_data.loc[nov_idx, 'late_rate'] * 100], color='black', s=120, zorder=5)
ax4.annotate('Black Friday 2017 Spike\n(14.2% Late Rate)', xy=(nov_idx, monthly_data.loc[nov_idx, 'late_rate'] * 100),
             xytext=(nov_idx - 3, 16), arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fff2f2", ec=colors['red']))

ax4.set_title('D. Seasonality Check: Black Friday Anomaly vs Normal Months', fontsize=12, fontweight='bold', color=colors['dark'])
ax4.set_ylabel('Late Delivery Rate (%)', fontsize=10)
ax4.legend(loc='upper left', frameon=True, fontsize=9)
ax4.set_ylim([0, 20])

plt.tight_layout()
plt.savefig('outputs/robustness_checks.png', dpi=300)
plt.close()

# Copy to brain artifact directory
artifact_dir = r'C:\Users\USER\.gemini\antigravity\brain\9c3f3cb4-a5ab-4607-a780-d83142b229e5'
if os.path.exists(artifact_dir):
    shutil.copy('outputs/robustness_checks.png', os.path.join(artifact_dir, 'robustness_checks.png'))

print("Robustness checks completed successfully! Saved outputs/robustness_checks.png, outputs/robustness_checks_summary.csv, outputs/robustness_delay_buckets.csv.")
