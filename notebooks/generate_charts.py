import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

os.makedirs('outputs', exist_ok=True)

# Load data
import duckdb
df = duckdb.connect().execute("SELECT * FROM 'outputs/olist_analytical_dataset.parquet'").df()
odds_q1 = pd.read_csv('outputs/q1_odds_ratios.csv')
feat_q1 = pd.read_csv('outputs/q1_feature_importance.csv')
routes = pd.read_csv('outputs/q1_top_risk_routes.csv')
cats = pd.read_csv('outputs/q1_top_risk_categories.csv')
odds_q2 = pd.read_csv('outputs/q2_odds_ratios.csv')
feat_q2 = pd.read_csv('outputs/q2_feature_importance.csv')

# -------------------------------------------------------------
# Chart 1: Q1 Feature Importance (Odds Ratios & Tree Importance)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Logistic Regression Odds Ratios
top_odds = odds_q1.head(8).copy()
colors_or = ['#d9534f' if x > 1.2 else '#f0ad4e' for x in top_odds['Odds_Ratio']]
y_pos = np.arange(len(top_odds))
ax1.barh(y_pos, top_odds['Odds_Ratio'], color=colors_or, edgecolor='black', alpha=0.85)
ax1.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Baseline (OR = 1.0)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_odds['Feature'], fontsize=11)
ax1.invert_yaxis()
ax1.set_xlabel('Odds Ratio (per Standard Deviation / Category)')
ax1.set_title('Top Risk Factors for Late Delivery\n(Logistic Regression Odds Ratios)')
for i, v in enumerate(top_odds['Odds_Ratio']):
    ax1.text(v + 0.02, i, f"{v:.2f}x", va='center', fontweight='bold', fontsize=10)
ax1.legend(loc='lower right')

# Subplot 2: Gradient Boosting Permutation Importance
top_perm = feat_q1.head(8).copy()
y_pos2 = np.arange(len(top_perm))
ax2.barh(y_pos2, top_perm['Importance'], color='#337ab7', edgecolor='black', alpha=0.85)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(top_perm['Feature'], fontsize=11)
ax2.invert_yaxis()
ax2.set_xlabel('Mean Decrease in ROC-AUC (Permutation Importance)')
ax2.set_title('Non-Linear Feature Importance for Late Delivery\n(HistGradientBoosting Classifier)')
for i, v in enumerate(top_perm['Importance']):
    ax2.text(v + 0.002, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/q1_feature_importance.png', dpi=300)
plt.close()
print("Saved outputs/q1_feature_importance.png")

# -------------------------------------------------------------
# Chart 2: Top High Risk Shipping Routes & Corridors
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

top_routes = routes.head(8).copy()
y_pos = np.arange(len(top_routes))
bars = ax1.barh(y_pos, top_routes['late_rate'] * 100, color='#e74c3c', edgecolor='black', alpha=0.85)
ax1.axvline(8.11, color='navy', linestyle='--', linewidth=1.5, label='National Baseline Lateness (8.11%)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{r} (N={n:,})" for r, n in zip(top_routes['route'], top_routes['order_count'])], fontsize=11)
ax1.invert_yaxis()
ax1.set_xlabel('Lateness Rate (%)')
ax1.set_title('Top 8 Highest-Risk Shipping Corridors (Min 100 Orders)')
for i, v in enumerate(top_routes['late_rate'] * 100):
    ax1.text(v + 0.5, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
ax1.legend(loc='lower right')

# Volume vs Lateness scatter for top states
state_stats = df.groupby('customer_state').agg(
    orders=('order_id', 'count'),
    late_rate=('is_late', 'mean'),
    low_review_rate=('is_low_review', 'mean')
).reset_index()

ax2.scatter(state_stats['late_rate'] * 100, state_stats['low_review_rate'] * 100, 
            s=state_stats['orders'] / 25, color='#2980b9', alpha=0.6, edgecolors='black', linewidth=1.5)
for _, row in state_stats.iterrows():
    if row['orders'] > 800 or row['late_rate'] > 0.15:
        ax2.annotate(row['customer_state'], (row['late_rate'] * 100 + 0.3, row['low_review_rate'] * 100 + 0.2), 
                     fontsize=10, fontweight='bold')

ax2.axvline(8.11, color='gray', linestyle=':', label='National Avg Lateness (8.11%)')
ax2.axhline(12.85, color='gray', linestyle='--', label='National Avg Low Review (12.85%)')
ax2.set_xlabel('Lateness Rate (%)')
ax2.set_ylabel('Low Review Rate (%) [Score <= 2]')
ax2.set_title('Customer State Delivery Risk vs. Low Review Risk\n(Bubble size = Order Volume)')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('outputs/q1_high_risk_routes.png', dpi=300)
plt.close()
print("Saved outputs/q1_high_risk_routes.png")

# -------------------------------------------------------------
# Chart 3: Fulfillment Bottlenecks Breakdown
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

stages = ['Payment Approval', 'Seller Dispatch Handling', 'Carrier Transit', 'Total Fulfillment']
ontime_vals = [0.34/24, 1.75, 6.92, 9.73]
late_vals = [0.41/24, 3.01, 23.92, 29.15]
x = np.arange(len(stages))
width = 0.35

rects1 = ax.bar(x - width/2, ontime_vals, width, label='On-Time Orders', color='#2ecc71', edgecolor='black')
rects2 = ax.bar(x + width/2, late_vals, width, label='Late Orders', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Duration (Calendar Days)')
ax.set_title('Fulfillment Stage Duration: On-Time vs. Late Orders (Median Days)')
ax.set_xticks(x)
ax.set_xticklabels(stages, fontweight='bold')
ax.legend(fontsize=12)

# Value annotations
for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}d", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}d", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

# Callout arrow for carrier transit
ax.annotate('Carrier Transit accounts for\n87.5% of total excess delay!',
            xy=(2.18, 24), xytext=(1.4, 26),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.6))

plt.tight_layout()
plt.savefig('outputs/q1_bottleneck_waterfall.png', dpi=300)
plt.close()
print("Saved outputs/q1_bottleneck_waterfall.png")

# -------------------------------------------------------------
# Chart 4: Q2 Negative Drivers of Customer Reviews
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

top_q2_odds = odds_q2.head(8).copy()
y_pos = np.arange(len(top_q2_odds))
ax1.barh(y_pos, top_q2_odds['Odds_Ratio'], color='#e67e22', edgecolor='black', alpha=0.85)
ax1.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Baseline (OR = 1.0)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_q2_odds['Feature'], fontsize=11)
ax1.invert_yaxis()
ax1.set_xlabel('Odds Ratio of Low Review (Score <= 2)')
ax1.set_title('Top Predictors of Low Customer Review\n(Logistic Regression Odds Ratios)')
for i, v in enumerate(top_q2_odds['Odds_Ratio']):
    ax1.text(v + 0.02, i, f"{v:.2f}x", va='center', fontweight='bold', fontsize=10)
ax1.legend(loc='lower right')

top_q2_perm = feat_q2.head(8).copy()
y_pos2 = np.arange(len(top_q2_perm))
ax2.barh(y_pos2, top_q2_perm['Importance'], color='#8e44ad', edgecolor='black', alpha=0.85)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(top_q2_perm['Feature'], fontsize=11)
ax2.invert_yaxis()
ax2.set_xlabel('Permutation Importance (Mean Decrease in ROC-AUC)')
ax2.set_title('Feature Importance for Low Review Classification\n(HistGradientBoosting Classifier)')
for i, v in enumerate(top_q2_perm['Importance']):
    ax2.text(v + 0.001, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/q2_feature_importance.png', dpi=300)
plt.close()
print("Saved outputs/q2_feature_importance.png")

# -------------------------------------------------------------
# Chart 5: Delay Days vs Customer Low Review Probability
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

df_delay = df[df['is_low_review'].notnull()].copy()
df_delay['delay_bin'] = pd.cut(
    df_delay['delay_days'],
    bins=[-50, -15, -10, -5, -2, 0, 2, 5, 10, 15, 25, 60],
    labels=['< -15d (Early)', '-15 to -10d', '-10 to -5d', '-5 to -2d', '-2 to 0d', 
            '0 to +2d (Late)', '+2 to +5d', '+5 to +10d', '+10 to +15d', '+15 to +25d', '> +25d']
)

delay_agg = df_delay.groupby('delay_bin', observed=False).agg(
    order_count=('order_id', 'count'),
    low_review_pct=('is_low_review', 'mean'),
    avg_score=('review_score', 'mean')
).reset_index()

delay_agg['low_review_pct'] = delay_agg['low_review_pct'] * 100.0

colors = ['#27ae60' if '-' in str(b) or '<' in str(b) else '#e74c3c' for b in delay_agg['delay_bin']]
bars = ax.bar(range(len(delay_agg)), delay_agg['low_review_pct'], color=colors, edgecolor='black', alpha=0.85)

ax.set_xticks(range(len(delay_agg)))
ax.set_xticklabels(delay_agg['delay_bin'], rotation=35, ha='right', fontsize=10)
ax.set_ylabel('Probability of Low Review (Score <= 2) [%]', fontsize=12)
ax.set_title('Non-Linear Relationship: Delivery Delay vs. Probability of Low Review', fontsize=13)
ax.axvline(4.5, color='black', linestyle='--', linewidth=2, label='Promised Delivery Date Threshold')

for i, v in enumerate(delay_agg['low_review_pct']):
    ax.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10)

ax.annotate('Threshold Effect:\nLateness triggers an immediate\n3x spike in low reviews!',
            xy=(5, 35), xytext=(2.5, 55),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#ffecb3", alpha=0.8))

ax.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/q2_delay_vs_review_score.png', dpi=300)
plt.close()
print("Saved outputs/q2_delay_vs_review_score.png")

# -------------------------------------------------------------
# Chart 6: Question 3 Attributable Risk & Low Review Distribution
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Low review rate comparison
categories = ['On-Time Deliveries\n(88,163 orders)', 'Late Deliveries\n(7,661 orders)']
rates = [9.26, 54.12]
bars = ax1.bar(categories, rates, color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5, alpha=0.85)
ax1.set_ylabel('Low Review Rate (%) [Score <= 2]', fontsize=12)
ax1.set_title('Low Review Rate by Delivery Performance\n(Relative Risk: 5.84x)', fontsize=13)
for b, r in zip(bars, rates):
    ax1.text(b.get_x() + b.get_width()/2, r + 1.5, f"{r:.2f}%", ha='center', fontweight='bold', fontsize=12)
ax1.set_ylim(0, 65)

# Attributable fraction donut chart
labels = ['Directly Attributable to\nDelivery Lateness (PAR)', 'Other Root Causes\n(Product Quality, Defect, Seller)']
sizes = [27.92, 72.08]
colors_pie = ['#e74c3c', '#3498db']
explode = (0.08, 0)
wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                   startangle=140, pctdistance=0.75, textprops=dict(fontsize=11, fontweight='bold'),
                                   wedgeprops=dict(width=0.45, edgecolor='black'))
ax2.set_title('Decomposition of All Marketplace Low Reviews\n(Population Attributable Risk = 27.92%)', fontsize=13)

plt.tight_layout()
plt.savefig('outputs/q3_attributable_risk_waterfall.png', dpi=300)
plt.close()
print("Saved outputs/q3_attributable_risk_waterfall.png")

# -------------------------------------------------------------
# Chart 7: Review Comment Topic Breakdown (Late vs On-time)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

topics = [
    'Delivery Delay / Transit Issue',
    'Customer Service / Communication',
    'Wrong Product / Not as Described',
    'Defective / Broken / Damaged',
    'Poor Quality / Dissatisfaction',
    'Missing Parts / Incomplete',
    'Other / General Complaint'
]
late_topic_pcts = [66.5, 10.0, 1.5, 0.9, 2.8, 0.6, 17.6]
ontime_topic_pcts = [38.2, 8.6, 8.5, 7.1, 6.9, 5.6, 25.1]

y = np.arange(len(topics))
height = 0.35

rects1 = ax.barh(y - height/2, ontime_topic_pcts, height, label='On-Time Low Reviews (Score <= 2)', color='#3498db', edgecolor='black')
rects2 = ax.barh(y + height/2, late_topic_pcts, height, label='Late Low Reviews (Score <= 2)', color='#e74c3c', edgecolor='black')

ax.set_yticks(y)
ax.set_yticklabels(topics, fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Percentage of Analyzed Review Comments (%)', fontsize=12)
ax.set_title('Root Cause Analysis: Review Comment Topics for Late vs. On-Time Low Reviews', fontsize=13)
ax.legend(loc='lower right', fontsize=11)

for r in rects1:
    w = r.get_width()
    ax.annotate(f"{w:.1f}%", xy=(w + 0.5, r.get_y() + r.get_height()/2), va='center', fontsize=9, fontweight='bold')
for r in rects2:
    w = r.get_width()
    ax.annotate(f"{w:.1f}%", xy=(w + 0.5, r.get_y() + r.get_height()/2), va='center', fontsize=9, fontweight='bold', color='#c0392b')

plt.tight_layout()
plt.savefig('outputs/q3_review_topic_breakdown.png', dpi=300)
plt.close()
print("Saved outputs/q3_review_topic_breakdown.png")

# -------------------------------------------------------------
# Chart 8: Actionable Recommendations Prioritization Matrix
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))

# Initiatives
initiatives = [
    {"name": "1. Postpone Review Survey Until Delivered", "effort": 1.5, "impact": 9.2, "category": "Platform UX"},
    {"name": "2. Route-Specific Dynamic Delivery SLA (SP->RJ & NE)", "effort": 3.8, "impact": 8.8, "category": "Logistics"},
    {"name": "3. Enforce Seller Dispatch SLA & Penalize Delays", "effort": 3.2, "impact": 7.5, "category": "Seller Mgmt"},
    {"name": "4. Regional Hub / Cross-Docking in RJ & NE", "effort": 8.5, "impact": 8.5, "category": "Logistics"},
    {"name": "5. Proactive 'Delayed in Transit' Notifications", "effort": 2.2, "impact": 6.8, "category": "Platform UX"},
    {"name": "6. Heavy / Bulky Item Specialized Freight Network", "effort": 7.0, "impact": 5.8, "category": "Logistics"},
    {"name": "7. Seller Quality Scoring & Throttling", "effort": 4.5, "impact": 7.2, "category": "Seller Mgmt"}
]

color_map = {"Platform UX": "#2ecc71", "Logistics": "#e74c3c", "Seller Mgmt": "#3498db"}

for init in initiatives:
    ax.scatter(init["effort"], init["impact"], s=450, color=color_map[init["category"]], 
               edgecolors='black', linewidth=1.5, alpha=0.85)
    ax.annotate(init["name"], (init["effort"] + 0.2, init["impact"] - 0.1), fontsize=10, fontweight='bold')

ax.axvline(5.0, color='gray', linestyle=':', alpha=0.7)
ax.axhline(6.5, color='gray', linestyle=':', alpha=0.7)

ax.text(2.0, 9.8, 'QUICK WINS\n(High Impact, Low Effort)', fontsize=12, fontweight='bold', color='#27ae60',
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#27ae60", alpha=0.7))
ax.text(7.0, 9.8, 'STRATEGIC BETS\n(High Impact, High Effort)', fontsize=12, fontweight='bold', color='#2980b9',
        bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#2980b9", alpha=0.7))
ax.text(2.0, 4.5, 'TACTICAL ADJUSTMENTS\n(Lower Impact, Low Effort)', fontsize=11, fontweight='bold', color='#7f8c8d')
ax.text(7.0, 4.5, 'DEPRIORITIZE\n(Lower Impact, High Effort)', fontsize=11, fontweight='bold', color='#7f8c8d')

ax.set_xlim(0.5, 10)
ax.set_ylim(4.0, 10.5)
ax.set_xlabel('Operational Effort / Complexity (1 = Low, 10 = High)', fontsize=12)
ax.set_ylabel('Customer Satisfaction & Retention Impact (1 = Low, 10 = High)', fontsize=12)
ax.set_title('Olist Strategic Operational Roadmap: Effort vs. Impact Prioritization Matrix', fontsize=14)

# Legend
for cat, col in color_map.items():
    ax.scatter([], [], s=120, c=col, edgecolors='black', label=cat)
ax.legend(loc='lower right', title='Initiative Domain', fontsize=10, title_fontsize=11)

plt.tight_layout()
plt.savefig('outputs/q4_recommendations_matrix.png', dpi=300)
plt.close()
print("Saved outputs/q4_recommendations_matrix.png")
print("\nAll 8 charts successfully generated in outputs/!")
