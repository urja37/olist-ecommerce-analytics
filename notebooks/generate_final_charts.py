import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os

os.makedirs('outputs', exist_ok=True)
artifact_dir = r'C:\Users\USER\.gemini\antigravity\brain\9c3f3cb4-a5ab-4607-a780-d83142b229e5'
os.makedirs(artifact_dir, exist_ok=True)

# Set global styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

palette = {
    'navy': '#1a365d',
    'blue': '#2b6cb0',
    'teal': '#319795',
    'green': '#2f855a',
    'red': '#c53030',
    'orange': '#dd6b20',
    'gray': '#718096',
    'light_gray': '#edf2f7'
}

# Load clean master data
print("Loading data from olist.duckdb (order_master)...")
con = duckdb.connect("olist.duckdb")
df = con.execute("SELECT * FROM order_master").df()
df_rev = df.dropna(subset=['review_score']).copy()

# =============================================================================
# Chart 1: The Rating Collapse (Review Score Distribution: On-Time vs. Late)
# Takeaway: Late Deliveries Trigger a 7x Surge in 1-Star Reviews and Erode 1.7 Stars
# =============================================================================
print("Generating Chart 1: Review Score Distribution...")
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

rev_crosstab = pd.crosstab(df_rev['review_score'], df_rev['is_late'], normalize='columns') * 100
x = np.arange(1, 6)
w = 0.38

bars_ontime = ax.bar(x - w/2, rev_crosstab[0], width=w, label='On-Time Deliveries (Mean: 4.29 Stars)', color=palette['green'], alpha=0.9, edgecolor='none')
bars_late = ax.bar(x + w/2, rev_crosstab[1], width=w, label='Late Deliveries (Mean: 2.56 Stars)', color=palette['red'], alpha=0.9, edgecolor='none')

# Add percentage labels above bars
for bar in bars_ontime:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha='center', fontsize=9.5, fontweight='bold', color=palette['green'])

for bar in bars_late:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha='center', fontsize=9.5, fontweight='bold', color=palette['red'])

# Annotate 1-star surge and 5-star drop
ax.annotate("7.0x Surge in 1-Star Ratings\n(6.6% -> 46.2%)",
            xy=(1 + w/2, 46.2), xytext=(1.5, 55),
            arrowprops=dict(facecolor=palette['red'], shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5, color=palette['red'],
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=palette['red'], lw=1.2))

ax.annotate("64% Collapse in 5-Star Ratings\n(62.4% -> 22.2%)",
            xy=(5 - w/2, 62.4), xytext=(3.6, 68),
            arrowprops=dict(facecolor=palette['green'], shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5, color=palette['green'],
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0fff4", ec=palette['green'], lw=1.2))

ax.set_title("Late Deliveries Trigger a 7x Surge in 1-Star Reviews and Erode 1.7 Stars on Average", fontsize=12.5, fontweight='bold', color=palette['navy'], pad=18)
ax.text(0.5, 1.02, "Comparing review score distributions for 87,544 on-time vs. 7,607 late delivered orders", transform=ax.transAxes, ha='center', fontsize=9.5, color=palette['gray'], style='italic')

ax.set_xlabel("Customer Review Score (Stars)", fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel("Percentage within Delivery Cohort (%)", fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(['1 Star\n(Very Dissatisfied)', '2 Stars\n(Dissatisfied)', '3 Stars\n(Neutral)', '4 Stars\n(Satisfied)', '5 Stars\n(Delighted)'], fontsize=9.5)
ax.set_ylim([0, 78])
ax.legend(loc='upper center', frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout()
plt.savefig('outputs/final_chart_1_rating_collapse.png', dpi=300)
shutil.copy('outputs/final_chart_1_rating_collapse.png', os.path.join(artifact_dir, 'final_chart_1_rating_collapse.png'))
plt.close()

# =============================================================================
# Chart 2: The Non-Linear Delay Cliff (Review Scores by Days Late)
# Takeaway: Customer Tolerance Collapses After 3 Days of Delay
# =============================================================================
print("Generating Chart 2: The Delay Cliff...")
fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)

def delay_group(r):
    if r['is_late'] == 0 or r['days_late'] <= 0:
        return 'On-Time\nBaseline'
    elif r['days_late'] <= 3:
        return '1 to 3\nDays Late'
    elif r['days_late'] <= 7:
        return '4 to 7\nDays Late'
    elif r['days_late'] <= 14:
        return '8 to 14\nDays Late'
    else:
        return '15+\nDays Late'

df_rev['delay_cat'] = df_rev.apply(delay_group, axis=1)
cat_order = ['On-Time\nBaseline', '1 to 3\nDays Late', '4 to 7\nDays Late', '8 to 14\nDays Late', '15+\nDays Late']
cat_stats = df_rev.groupby('delay_cat', observed=False).agg(
    orders=('order_id', 'count'),
    avg_score=('review_score', 'mean'),
    low_rate=('low_review', 'mean')
).loc[cat_order].reset_index()

x_cats = np.arange(len(cat_order))
bar_colors = [palette['green'], palette['orange'], palette['red'], '#9b2c2c', '#742a2a']

bars = ax1.bar(x_cats, cat_stats['avg_score'], width=0.55, color=bar_colors, alpha=0.85, edgecolor='none')
for bar in bars:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + 0.1, f"{y:.2f} stars", ha='center', fontweight='bold', fontsize=9.5)

# Add line for low review rate on second axis
ax2 = ax1.twinx()
ax2.plot(x_cats, cat_stats['low_rate'] * 100, color=palette['navy'], marker='o', lw=2.5, markersize=8, label='Low Review Rate (<= 2 Stars)')
for i, txt in enumerate(cat_stats['low_rate']):
    ax2.annotate(f"{txt:.1%}", (x_cats[i], txt * 100), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color=palette['navy'], fontsize=9.5)

ax1.axhline(4.29, color='gray', linestyle='--', lw=1.2, alpha=0.7)
ax1.set_title("Customer Tolerance Collapses After 3 Days of Delay", fontsize=12.5, fontweight='bold', color=palette['navy'], pad=18)
ax1.text(0.5, 1.02, "Minor 1-3 day delays drop ratings by 0.5 stars; delays of 4+ days breach customer tolerance and collapse ratings", transform=ax1.transAxes, ha='center', fontsize=9.5, color=palette['gray'], style='italic')

ax1.set_xlabel("Delivery Status & Delay Duration", fontsize=10.5, fontweight='bold', labelpad=8)
ax1.set_ylabel("Average Review Score (1 to 5 Stars)", fontsize=10.5, fontweight='bold', labelpad=8)
ax2.set_ylabel("Percentage of Low Reviews (<= 2 Stars)", fontsize=10.5, fontweight='bold', color=palette['navy'], labelpad=8)
ax1.set_xticks(x_cats)
ax1.set_xticklabels(cat_order, fontsize=9.5)
ax1.set_ylim([0, 5.5])
ax2.set_ylim([0, 85])
ax2.grid(False)

# Cliff annotation
ax1.annotate("The Tolerance Cliff:\nRating drops from 3.76 -> 2.32 stars\nLow review rate doubles (31% -> 60%)",
             xy=(2, 2.32), xytext=(2.2, 3.8),
             arrowprops=dict(facecolor=palette['red'], shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', fontsize=9, color=palette['red'],
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=palette['red'], lw=1.2))

plt.tight_layout()
plt.savefig('outputs/final_chart_2_delay_cliff.png', dpi=300)
shutil.copy('outputs/final_chart_2_delay_cliff.png', os.path.join(artifact_dir, 'final_chart_2_delay_cliff.png'))
plt.close()

# =============================================================================
# Chart 3: Order Fulfillment Lifecycle Decomposition
# Takeaway: Carrier Transit Accounts for 87.5% of Total Delivery Delay
# =============================================================================
print("Generating Chart 3: Lifecycle Bottlenecks...")
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

stages = ['Payment Approval', 'Seller Warehouse Handling', 'Carrier Transit to Customer']
ontime_means = [
    (pd.to_datetime(df[df['is_late']==0]['order_approved_at']) - pd.to_datetime(df[df['is_late']==0]['order_purchase_timestamp'])).dt.total_seconds().mean() / 86400,
    df[df['is_late']==0]['seller_handling_days'].mean(),
    (pd.to_datetime(df[df['is_late']==0]['order_delivered_customer_date']) - pd.to_datetime(df[df['is_late']==0]['order_delivered_carrier_date'])).dt.total_seconds().mean() / 86400
]

late_means = [
    (pd.to_datetime(df[df['is_late']==1]['order_approved_at']) - pd.to_datetime(df[df['is_late']==1]['order_purchase_timestamp'])).dt.total_seconds().mean() / 86400,
    df[df['is_late']==1]['seller_handling_days'].mean(),
    (pd.to_datetime(df[df['is_late']==1]['order_delivered_customer_date']) - pd.to_datetime(df[df['is_late']==1]['order_delivered_carrier_date'])).dt.total_seconds().mean() / 86400
]

y_pos = np.arange(len(stages))
h = 0.35

rects1 = ax.barh(y_pos - h/2, ontime_means, height=h, label='On-Time Deliveries (Total: 8.2 Days)', color=palette['green'], alpha=0.85)
rects2 = ax.barh(y_pos + h/2, late_means, height=h, label='Late Deliveries (Total: 26.5 Days)', color=palette['red'], alpha=0.85)

for rect in rects1:
    w_val = rect.get_width()
    ax.text(w_val + 0.4, rect.get_y() + rect.get_height()/2, f"{w_val:.1f}d", va='center', fontweight='bold', color=palette['green'], fontsize=9.5)

for rect in rects2:
    w_val = rect.get_width()
    ax.text(w_val + 0.4, rect.get_y() + rect.get_height()/2, f"{w_val:.1f}d", va='center', fontweight='bold', color=palette['red'], fontsize=9.5)

ax.set_title("Carrier Transit Accounts for 87.5% of Total Delivery Delay (+17.0 Days)", fontsize=12.5, fontweight='bold', color=palette['navy'], pad=18)
ax.text(0.5, 1.02, "Decomposing fulfillment lifecycle days: Approval is negligible and seller handling adds +1.26d, but carrier transit explodes", transform=ax.transAxes, ha='center', fontsize=9.5, color=palette['gray'], style='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels(stages, fontsize=10, fontweight='bold')
ax.set_xlabel("Average Stage Duration (Days)", fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_xlim([0, 27])
ax.legend(loc='lower right', frameon=True, fontsize=9.5)

# Add callout on transit bar
ax.annotate("+17.0 Days Transit Surge\n(Explains 87.5% of overall order delay)",
            xy=(23.9, 2 + h/2), xytext=(16, 1.3),
            arrowprops=dict(facecolor=palette['red'], shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5, color=palette['red'],
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=palette['red'], lw=1.2))

plt.tight_layout()
plt.savefig('outputs/final_chart_3_bottleneck_waterfall.png', dpi=300)
shutil.copy('outputs/final_chart_3_bottleneck_waterfall.png', os.path.join(artifact_dir, 'final_chart_3_bottleneck_waterfall.png'))
plt.close()

# =============================================================================
# Chart 4: Geographic Polarization & The SP -> RJ Hotspot
# Takeaway: Shipping to Rio de Janeiro Multiplies Delay Odds by 3.7x and Leaks Over 1,200 Orders
# =============================================================================
print("Generating Chart 4: Geographic Hotspots...")
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

routes_summary = [
    {'Route': 'SP -> SP (Intrastate Local)', 'Orders': 39500, 'Late_Rate': 5.22, 'Group': 'Local Benchmark'},
    {'Route': 'SP -> PR (South Regional)', 'Orders': 4580, 'Late_Rate': 5.48, 'Group': 'Fast Regional Hub'},
    {'Route': 'SP -> MG (Southeast Regional)', 'Orders': 9820, 'Late_Rate': 7.64, 'Group': 'Fast Regional Hub'},
    {'Route': 'SP -> DF (Central Capital)', 'Orders': 1850, 'Late_Rate': 6.81, 'Group': 'Central Corridor'},
    {'Route': 'SP -> RJ (Metro Bottleneck)', 'Orders': 8150, 'Late_Rate': 15.52, 'Group': 'Severe Bottleneck'},
    {'Route': 'SP -> BA (Northeast Hub)', 'Orders': 2450, 'Late_Rate': 15.10, 'Group': 'Severe Bottleneck'},
    {'Route': 'SP -> CE (Northeast Long-Haul)', 'Orders': 920, 'Late_Rate': 17.83, 'Group': 'Severe Bottleneck'},
    {'Route': 'SP -> MA (North Long-Haul)', 'Orders': 530, 'Late_Rate': 21.32, 'Group': 'Severe Bottleneck'},
    {'Route': 'SP -> AL (Northeast Deep)', 'Orders': 280, 'Late_Rate': 26.07, 'Group': 'Severe Bottleneck'}
]
rdf = pd.DataFrame(routes_summary).sort_values('Late_Rate', ascending=True)

y_r = np.arange(len(rdf))
colors_r = [palette['green'] if r < 8.11 else (palette['orange'] if r < 12 else palette['red']) for r in rdf['Late_Rate']]

bars_r = ax.barh(y_r, rdf['Late_Rate'], color=colors_r, height=0.6, alpha=0.85)
for i, (bar, row) in enumerate(zip(bars_r, rdf.itertuples())):
    w_val = bar.get_width()
    ax.text(w_val + 0.4, bar.get_y() + bar.get_height()/2, f"{w_val:.1f}% ({row.Orders:,} orders)", va='center', fontweight='bold', fontsize=8.5, color='#333')

ax.axvline(8.11, color=palette['navy'], linestyle='--', lw=1.5, label='Marketplace Average Late Rate (8.11%)')

ax.set_title("Shipping to Rio de Janeiro Multiplies Delay Odds by 3.7x and Leaks Over 1,200 Orders", fontsize=12, fontweight='bold', color=palette['navy'], pad=18)
ax.text(0.5, 1.02, "Comparing late delivery rates across major shipping lanes from São Paulo sellers", transform=ax.transAxes, ha='center', fontsize=9.5, color=palette['gray'], style='italic')

ax.set_yticks(y_r)
ax.set_yticklabels(rdf['Route'], fontsize=9.5)
ax.set_xlabel("Late Delivery Rate (%)", fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_xlim([0, 31])
ax.legend(loc='lower right', frameon=True, fontsize=9.5)

# Annotation for SP -> RJ
rj_idx = rdf[rdf['Route'].str.contains('SP -> RJ')].index[0]
ax.annotate("SP -> RJ Volume Hotspot:\n15.5% late rate across 8,150 orders\nSingle largest delay leakage on Olist",
            xy=(15.52, 4), xytext=(18, 2.2),
            arrowprops=dict(facecolor=palette['red'], shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9, color=palette['red'],
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=palette['red'], lw=1.2))

plt.tight_layout()
plt.savefig('outputs/final_chart_4_geographic_hotspots.png', dpi=300)
shutil.copy('outputs/final_chart_4_geographic_hotspots.png', os.path.join(artifact_dir, 'final_chart_4_geographic_hotspots.png'))
plt.close()

# =============================================================================
# Chart 5: Sizing the Levers (Odds Ratios Forest Plot)
# Takeaway: Seller Handling and Distance Double Delay Odds, While Delivery Buffer Cuts Risk by 73%
# =============================================================================
print("Generating Chart 5: Odds Ratios Forest Plot...")
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)

forest_data = [
    {'Label': 'Late Delivery on Low Review (Model B)', 'OR': 11.24, 'Low': 10.58, 'High': 11.93, 'Type': 'Bad Review Driver'},
    {'Label': 'Shipping to Rio de Janeiro vs SP (Model A)', 'OR': 3.73, 'Low': 3.22, 'High': 4.31, 'Type': 'Late Delivery Driver'},
    {'Label': 'Seller Handling Days (per 1 SD increase)', 'OR': 2.22, 'Low': 2.17, 'High': 2.28, 'Type': 'Late Delivery Driver'},
    {'Label': 'Shipping Distance (log km, per 1 SD increase)', 'OR': 1.50, 'Low': 1.42, 'High': 1.58, 'Type': 'Late Delivery Driver'},
    {'Label': 'Multiple Items in Cart (per 1 SD increase)', 'OR': 1.25, 'Low': 1.22, 'High': 1.28, 'Type': 'Bad Review Driver'},
    {'Label': 'Same-State Fulfillment (Local Delivery)', 'OR': 0.61, 'Low': 0.54, 'High': 0.69, 'Type': 'Protective Factor'},
    {'Label': 'Estimated Delivery Window (per 1 SD increase)', 'OR': 0.27, 'Low': 0.25, 'High': 0.28, 'Type': 'Protective Factor'}
]
fdf = pd.DataFrame(forest_data).iloc[::-1].reset_index(drop=True)

y_f = np.arange(len(fdf))
colors_f = [palette['green'] if or_val < 1.0 else palette['red'] for or_val in fdf['OR']]

ax.axvline(1.0, color='gray', linestyle='--', lw=1.5, zorder=1)
for i, row in fdf.iterrows():
    c = palette['green'] if row['OR'] < 1.0 else palette['red']
    ax.errorbar(row['OR'], i, xerr=[[row['OR'] - row['Low']], [row['High'] - row['OR']]],
                fmt='o', color=c, ecolor=c, elinewidth=2.5, capsize=5, markersize=8, zorder=3)

for i, row in fdf.iterrows():
    color_txt = palette['green'] if row['OR'] < 1.0 else palette['red']
    ax.text(row['OR'] * 1.15 if row['OR'] > 1.0 else row['OR'] * 0.70, i, f"OR = {row['OR']:.2f}x",
            va='center', ha='left' if row['OR'] > 1.0 else 'right', fontweight='bold', fontsize=9.5, color=color_txt)

ax.set_title("Seller Handling and Distance Double Delay Odds, While Delivery Buffer Cuts Risk by 73%", fontsize=12, fontweight='bold', color=palette['navy'], pad=18)
ax.text(0.5, 1.02, "Standardized Odds Ratios with 95% Confidence Intervals from Models A & B (Logarithmic Scale)", transform=ax.transAxes, ha='center', fontsize=9.5, color=palette['gray'], style='italic')

ax.set_yticks(y_f)
ax.set_yticklabels(fdf['Label'], fontsize=9.5, fontweight='bold')
ax.set_xlabel("Odds Ratio (exp(Beta)) - Values > 1 increase risk, Values < 1 reduce risk", fontsize=10, fontweight='bold', labelpad=8)
ax.set_xscale('log')
ax.set_xlim([0.15, 20])
ax.grid(True, which='both', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('outputs/final_chart_5_odds_ratios_levers.png', dpi=300)
shutil.copy('outputs/final_chart_5_odds_ratios_levers.png', os.path.join(artifact_dir, 'final_chart_5_odds_ratios_levers.png'))
plt.close()

# =============================================================================
# Chart 6: 4-Panel Executive Dashboard (Unified View)
# =============================================================================
print("Generating Executive Summary Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Panel 1: Rating Collapse
axes[0, 0].bar(x - w/2, rev_crosstab[0], width=w, label='On-Time (4.29 Stars)', color=palette['green'], alpha=0.9)
axes[0, 0].bar(x + w/2, rev_crosstab[1], width=w, label='Late (2.56 Stars)', color=palette['red'], alpha=0.9)
for bar in axes[0, 0].patches:
    h = bar.get_height()
    if h > 2:
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.0f}%", ha='center', fontsize=8.5, fontweight='bold')
axes[0, 0].set_title("A. Late Deliveries Cause a 7x Surge in 1-Star Reviews", fontsize=11, fontweight='bold', color=palette['navy'])
axes[0, 0].set_xlabel("Review Score (Stars)")
axes[0, 0].set_ylabel("Share of Orders (%)")
axes[0, 0].set_xticks(x)
axes[0, 0].legend()

# Panel 2: The Delay Cliff
bars2 = axes[0, 1].bar(x_cats, cat_stats['avg_score'], width=0.55, color=bar_colors, alpha=0.85)
for bar in bars2:
    y = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, y + 0.1, f"{y:.2f}", ha='center', fontweight='bold', fontsize=9)
axes[0, 1].set_title("B. Customer Tolerance Collapses After 3 Days of Delay", fontsize=11, fontweight='bold', color=palette['navy'])
axes[0, 1].set_xlabel("Delay Duration")
axes[0, 1].set_ylabel("Average Review Score (1 to 5 Stars)")
axes[0, 1].set_xticks(x_cats)
axes[0, 1].set_xticklabels(['On-Time', '1-3 Days', '4-7 Days', '8-14 Days', '15+ Days'])
axes[0, 1].set_ylim([0, 5.2])

# Panel 3: Lifecycle Breakdown
axes[1, 0].barh(y_pos - h/2, ontime_means, height=h, label='On-Time (8.2d)', color=palette['green'], alpha=0.85)
axes[1, 0].barh(y_pos + h/2, late_means, height=h, label='Late (26.5d)', color=palette['red'], alpha=0.85)
axes[1, 0].set_title("C. Carrier Transit Accounts for 87.5% of Fulfillment Delay", fontsize=11, fontweight='bold', color=palette['navy'])
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(['Approval', 'Handling', 'Transit'])
axes[1, 0].set_xlabel("Average Days")
axes[1, 0].legend(loc='lower right')

# Panel 4: Hotspots
bars_r2 = axes[1, 1].barh(y_r, rdf['Late_Rate'], color=colors_r, height=0.6, alpha=0.85)
axes[1, 1].axvline(8.11, color=palette['navy'], linestyle='--', lw=1.2, label='Avg (8.1%)')
axes[1, 1].set_title("D. Rio de Janeiro and Northeast Corridors Drive Lateness", fontsize=11, fontweight='bold', color=palette['navy'])
axes[1, 1].set_yticks(y_r)
axes[1, 1].set_yticklabels(rdf['Route'], fontsize=8.5)
axes[1, 1].set_xlabel("Late Rate (%)")
axes[1, 1].legend(loc='lower right')

plt.suptitle("Olist Operational Analytics: Core Empirical Findings & Policy Levers", fontsize=15, fontweight='bold', color=palette['navy'], y=0.99)
plt.tight_layout()
plt.savefig('outputs/final_executive_summary_dashboard.png', dpi=300)
shutil.copy('outputs/final_executive_summary_dashboard.png', os.path.join(artifact_dir, 'final_executive_summary_dashboard.png'))
plt.close()

print("\nSuccessfully generated all 5 standalone takeaway charts and the unified executive dashboard!")
print("Saved to outputs/ and copied to artifact directory.")
