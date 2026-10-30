import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

os.makedirs('outputs', exist_ok=True)
con = duckdb.connect('olist.duckdb')
df = con.execute("SELECT * FROM order_master").df()

overall_late_rate = df['is_late'].mean() * 100.0
print(f"Overall clean late rate: {overall_late_rate:.2f}%")

# =============================================================================
# Chart 1: Late Rate by Customer State
# Question: Which destination states face the highest delivery delay risks?
# =============================================================================
plt.figure(figsize=(14, 7))
state_df = df.groupby('customer_state').agg(
    orders=('order_id', 'count'),
    late_rate=('is_late', lambda x: x.mean() * 100.0)
).sort_values('late_rate', ascending=False).reset_index()

colors_state = ['#d9534f' if r > overall_late_rate else '#5cb85c' for r in state_df['late_rate']]
bars = plt.bar(state_df['customer_state'], state_df['late_rate'], color=colors_state, edgecolor='black', alpha=0.85)

plt.axhline(overall_late_rate, color='navy', linestyle='--', linewidth=2, 
            label=f'National Average Lateness ({overall_late_rate:.2f}%)')

plt.title('Late Delivery Rate by Customer Destination State\n(Red = Above National Average | Green = Below Average)', pad=15)
plt.xlabel('Customer State (UF)', fontweight='bold')
plt.ylabel('Late Delivery Rate (%)', fontweight='bold')
plt.xticks(rotation=0, fontweight='bold')
plt.ylim(0, 32)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, rotation=45, fontweight='bold')

plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/late_rate_by_state.png', dpi=300)
plt.close()
print("Saved outputs/late_rate_by_state.png")

# =============================================================================
# Chart 2: Late Rate by Distance Bucket
# Question: How does geographic distance between buyer and seller affect lateness?
# =============================================================================
plt.figure(figsize=(10, 6))
df['dist_bucket'] = pd.cut(
    df['distance_km'],
    bins=[-1, 100, 500, 1000, 10000],
    labels=['0–100 km (Local)', '100–500 km (Regional)', '500–1,000 km (Inter-Regional)', '1,000+ km (Cross-Country)']
)

dist_df = df.groupby('dist_bucket', observed=False).agg(
    orders=('order_id', 'count'),
    late_rate=('is_late', lambda x: x.mean() * 100.0)
).reset_index()

colors_dist = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
bars = plt.bar(dist_df['dist_bucket'], dist_df['late_rate'], color=colors_dist, edgecolor='black', width=0.55, alpha=0.85)
plt.axhline(overall_late_rate, color='gray', linestyle=':', linewidth=1.5, label=f'Overall Average ({overall_late_rate:.2f}%)')

plt.title('Delivery Lateness Rate by Shipping Distance Bucket\n(Distance escalates delay risk by ~85%)', pad=15)
plt.xlabel('Haversine Distance Bucket (km)', fontweight='bold')
plt.ylabel('Late Delivery Rate (%)', fontweight='bold')
plt.ylim(0, 15)

for bar, row in zip(bars, dist_df.itertuples()):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, 
             f'{yval:.2f}%\n(N={row.orders:,})', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/late_rate_by_distance.png', dpi=300)
plt.close()
print("Saved outputs/late_rate_by_distance.png")

# =============================================================================
# Chart 3: Late Rate by Month (Seasonality & Nov 2017 Spike)
# Question: What seasonal shocks and logistics bottlenecks occurred over time?
# =============================================================================
plt.figure(figsize=(14, 6))
df['year_month'] = pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M').astype(str)
month_df = df[(df['year_month'] >= '2017-01') & (df['year_month'] <= '2018-08')].groupby('year_month').agg(
    orders=('order_id', 'count'),
    late_rate=('is_late', lambda x: x.mean() * 100.0)
).reset_index()

plt.plot(month_df['year_month'], month_df['late_rate'], marker='o', linewidth=2.5, color='#c0392b', label='Monthly Late Rate (%)')
plt.axhline(overall_late_rate, color='navy', linestyle='--', linewidth=1.5, label=f'Historical Average ({overall_late_rate:.2f}%)')

# Callout 1: Black Friday Nov 2017
nov_idx = month_df[month_df['year_month'] == '2017-11'].index[0]
nov_val = month_df.loc[nov_idx, 'late_rate']
plt.annotate(f'Nov 2017 Black Friday Spike\n{nov_val:.1f}% Late\n(Order volume surge)',
             xy=(nov_idx, nov_val), xytext=(nov_idx - 1.8, nov_val + 4.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffecb3", ec="#f39c12"))

# Callout 2: March 2018 Correios Strike
mar_idx = month_df[month_df['year_month'] == '2018-03'].index[0]
mar_val = month_df.loc[mar_idx, 'late_rate']
plt.annotate(f'March 2018 Correios Strike\n{mar_val:.1f}% Late\n(National Postal Disruption)',
             xy=(mar_idx, mar_val), xytext=(mar_idx - 1.5, mar_val + 3.0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#f8d7da", ec="#e74c3c"))

plt.title('Monthly Delivery Lateness Rate (Jan 2017 – Aug 2018)\nIdentifying Macro Logistical Shocks & Seasonal Vulnerabilities', pad=15)
plt.xlabel('Order Purchase Month', fontweight='bold')
plt.ylabel('Late Delivery Rate (%)', fontweight='bold')
plt.xticks(rotation=45, fontweight='bold')
plt.ylim(0, 26)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/late_rate_by_month.png', dpi=300)
plt.close()
print("Saved outputs/late_rate_by_month.png")

# =============================================================================
# Chart 4: Review Score Distribution: On-Time vs. Late (Side-by-Side Bars)
# Question: How drastically does customer satisfaction collapse when delivery is late?
# =============================================================================
plt.figure(figsize=(12, 6.5))
df_rev = df[df['review_score'].notnull()]
ct = pd.crosstab(df_rev['review_score'], df_rev['is_late'], normalize='columns') * 100.0

x = np.arange(1, 6)
width = 0.38

rects1 = plt.bar(x - width/2, ct[0], width, label='On-Time Deliveries (N=88,085)', color='#2ecc71', edgecolor='black', alpha=0.9)
rects2 = plt.bar(x + width/2, ct[1], width, label='Late Deliveries (N=7,710)', color='#e74c3c', edgecolor='black', alpha=0.9)

plt.title('Customer Review Score Distribution: On-Time vs. Late Deliveries\n(The Collapse of Customer Delight upon Breach of SLA)', pad=15)
plt.xlabel('Review Score (Stars)', fontweight='bold', fontsize=12)
plt.ylabel('Percentage of Cohort (%)', fontweight='bold', fontsize=12)
plt.xticks(x, ['1 Star\n(Angry)', '2 Stars\n(Unhappy)', '3 Stars\n(Neutral)', '4 Stars\n(Satisfied)', '5 Stars\n(Delighted)'], fontsize=11, fontweight='bold')
plt.ylim(0, 75)

for r in rects1:
    h = r.get_height()
    plt.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for r in rects2:
    h = r.get_height()
    plt.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

# Callout comparing 1-star reviews
plt.annotate('1-Star reviews surge 7.0x\n(6.6% -> 46.2%)\nwhen delivery is late!',
             xy=(1.19, 47), xytext=(1.45, 58),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#ffecb3", ec="#e67e22"))

# Callout comparing 5-star reviews
plt.annotate('5-Star reviews drop by 65%\n(62.4% -> 22.2%)',
             xy=(4.81, 23), xytext=(3.5, 36),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#ebf5fb", ec="#2980b9"))

plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/review_score_ontime_vs_late.png', dpi=300)
plt.close()
print("Saved outputs/review_score_ontime_vs_late.png")

# =============================================================================
# Chart 5: Average Review Score by Freight Ratio Bucket
# Question: Does high freight cost relative to product price depress satisfaction?
# =============================================================================
plt.figure(figsize=(11, 6))
df['freight_ratio_bucket'] = pd.cut(
    df['freight_ratio'],
    bins=[-1, 0.10, 0.20, 0.30, 0.50, 1.0, 100],
    labels=['< 10%', '10%–20%', '20%–30%', '30%–50%', '50%–100%', '> 100% (Freight > Price)']
)

fr_df = df.groupby('freight_ratio_bucket', observed=False).agg(
    orders=('order_id', 'count'),
    avg_review=('review_score', 'mean'),
    low_review_pct=('low_review', lambda x: x.mean() * 100.0)
).reset_index()

fig, ax1 = plt.subplots(figsize=(11, 6))

color_bar = '#3498db'
bars = ax1.bar(fr_df['freight_ratio_bucket'], fr_df['avg_review'], color=color_bar, edgecolor='black', width=0.55, alpha=0.85)
ax1.set_ylabel('Average Review Score (1–5 Stars)', fontweight='bold', color='#1f618d', fontsize=12)
ax1.set_ylim(3.8, 4.4)
ax1.set_xlabel('Freight-to-Price Ratio Bucket', fontweight='bold', fontsize=12)

# Secondary axis for Low Review Rate (%)
ax2 = ax1.twinx()
ax2.plot(fr_df['freight_ratio_bucket'], fr_df['low_review_pct'], color='#e74c3c', marker='s', linewidth=2.5, markersize=8, label='Low Review Rate (%)')
ax2.set_ylabel('Low Review Rate (% [Score <= 2])', fontweight='bold', color='#c0392b', fontsize=12)
ax2.set_ylim(10, 18)
ax2.grid(False)

plt.title('Customer Satisfaction by Freight-to-Price Ratio Bucket\n(Higher relative freight fees steadily depress review ratings)', pad=15)

for bar, r in zip(bars, fr_df.itertuples()):
    h = bar.get_height()
    ax1.annotate(f"{h:.2f} stars\n(N={r.orders:,})", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for i, v in enumerate(fr_df['low_review_pct']):
    ax2.annotate(f"{v:.1f}%", xy=(i, v), xytext=(0, 7), textcoords="offset points",
                 ha='center', va='bottom', color='#c0392b', fontsize=10, fontweight='bold')

ax2.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/avg_review_by_freight_ratio.png', dpi=300)
plt.close()
print("Saved outputs/avg_review_by_freight_ratio.png")

# =============================================================================
# Chart 6: Correlation Heatmap of Numeric Features
# Question: What are the inter-correlations between fulfillment, commercial, and review features?
# =============================================================================
plt.figure(figsize=(13, 10))
corr_features = [
    'review_score', 'low_review', 'is_late', 'days_late', 'actual_days', 'estimated_days',
    'seller_handling_days', 'distance_km', 'freight_ratio_capped', 'total_price',
    'total_freight', 'weight_capped', 'volume_capped', 'n_items'
]

rename_dict = {
    'review_score': 'Review Score',
    'low_review': 'Low Review Flag',
    'is_late': 'Late Delivery Flag',
    'days_late': 'Days Late',
    'actual_days': 'Actual Days',
    'estimated_days': 'Estimated Days',
    'seller_handling_days': 'Seller Handling Days',
    'distance_km': 'Distance (km)',
    'freight_ratio_capped': 'Freight Ratio',
    'total_price': 'Total Price',
    'total_freight': 'Total Freight',
    'weight_capped': 'Weight (g)',
    'volume_capped': 'Volume (cm3)',
    'n_items': 'Item Count'
}

corr_matrix = df[corr_features].rename(columns=rename_dict).corr(method='spearman')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-0.4, vmax=0.7, annot=True, 
            fmt='.2f', square=True, linewidths=.8, cbar_kws={"shrink": .8})

plt.title('Spearman Correlation Heatmap: Operational & Commercial Drivers of Review Scores', pad=20)
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=300)
plt.close()
print("Saved outputs/correlation_heatmap.png")

print("\nAll 6 requested charts generated successfully in outputs/!")
