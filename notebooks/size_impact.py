import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect("olist.duckdb")
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
gap_z = mean_ontime - mean_late

rate_low_ontime = df[df['is_late'] == 0]['low_review'].mean()
rate_low_late = df[df['is_late'] == 1]['low_review'].mean()

print(f"Total orders with reviews: {n_total:,}")
print(f"Late orders: {n_late:,} ({pct_late:.1%})")
print(f"Total low reviews (1-2 stars): {n_low:,} ({pct_low:.1%})")
print(f"Late low reviews: {n_late_low:,}")
print(f"Late share of low reviews (Y%): {pct_late_in_low:.1%}")
print(f"Mean review score - On-time: {mean_ontime:.2f} stars")
print(f"Mean review score - Late:    {mean_late:.2f} stars")
print(f"Gap (Z):                     {gap_z:.2f} stars lower")

# State-level breakdown
state_stats = df.groupby('customer_state').agg(
    orders=('order_id', 'count'),
    late_orders=('is_late', 'sum'),
    late_rate=('is_late', 'mean')
).sort_values('late_rate', ascending=False)

print("\n--- TOP 10 STATES BY LATE RATE ---")
print(state_stats.head(10).to_string())

# Define worst states: States with late rate >= 12% (AL, MA, SE, PI, RR, PB, BA, CE, PA, RJ)
worst_states_12 = state_stats[state_stats['late_rate'] >= 0.12].index.tolist()
worst_df_12 = df[df['customer_state'].isin(worst_states_12)]
n_orders_worst = len(worst_df_12)
n_late_worst = int(worst_df_12['is_late'].sum())

# Scenario 1: Cut lateness by 1/3 in states with >= 12% late rate (includes RJ, BA, AL, MA, etc.)
late_cut_1 = n_late_worst * (1/3)
low_avoided_1 = late_cut_1 * (rate_low_late - rate_low_ontime)
pct_low_drop_1 = low_avoided_1 / n_low
score_gain_1 = (late_cut_1 * gap_z) / n_total

print("\n--- SCENARIO 1: Cut lateness by 1/3 across all high-delay states (late rate >= 12%, N=10 states including RJ) ---")
print(f"High-delay states: {', '.join(worst_states_12)}")
print(f"Total orders in these states: {n_orders_worst:,} | Late orders: {n_late_worst:,} ({n_late_worst/n_orders_worst:.1%})")
print(f"Late orders eliminated (1/3): {late_cut_1:,.0f}")
print(f"Low reviews avoided:          {low_avoided_1:,.0f}")
print(f"Marketplace low reviews drop (V%): {pct_low_drop_1:.1%}")
print(f"Marketplace avg review increase (W): +{score_gain_1:.3f} points (from {df['review_score'].mean():.2f} to {df['review_score'].mean() + score_gain_1:.2f})")

# Scenario 2: What about the worst 5 states by late rate (AL, MA, SE, PI, RR)?
top5_worst = state_stats.head(5).index.tolist()
top5_df = df[df['customer_state'].isin(top5_worst)]
late_cut_top5 = top5_df['is_late'].sum() * (1/3)
low_avoided_top5 = late_cut_top5 * (rate_low_late - rate_low_ontime)
pct_low_drop_top5 = low_avoided_top5 / n_low
score_gain_top5 = (late_cut_top5 * gap_z) / n_total

print("\n--- SCENARIO 2: Cut lateness by 1/3 in the 5 worst states by late rate (AL, MA, SE, PI, RR) ---")
print(f"States: {', '.join(top5_worst)}")
print(f"Late orders eliminated (1/3): {late_cut_top5:,.0f}")
print(f"Low reviews avoided:          {low_avoided_top5:,.0f}")
print(f"Marketplace low reviews drop (V%): {pct_low_drop_top5:.1%}")
print(f"Marketplace avg review increase (W): +{score_gain_top5:.3f} points")

# Scenario 3: What if we cut lateness by 1/3 across the top 3 biggest volume leak states (RJ, SP, MG)?
# RJ alone has 1,848 late orders, SP has 2,168, MG has 786.
# If we do just RJ + SP + MG:
top3_vol = ['RJ', 'BA', 'MG'] # Or RJ alone
rj_late = df[df['customer_state'] == 'RJ']['is_late'].sum()
late_cut_rj = rj_late * (1/3)
low_avoided_rj = late_cut_rj * (rate_low_late - rate_low_ontime)
pct_low_drop_rj = low_avoided_rj / n_low
score_gain_rj = (late_cut_rj * gap_z) / n_total

print("\n--- SCENARIO 3: Just Rio de Janeiro (RJ) Cut Lateness by 1/3 ---")
print(f"RJ late orders: {rj_late:,}")
print(f"Late orders eliminated (1/3): {late_cut_rj:,.0f}")
print(f"Low reviews avoided:          {low_avoided_rj:,.0f}")
print(f"Marketplace low reviews drop (V%): {pct_low_drop_rj:.1%}")
print(f"Marketplace avg review increase (W): +{score_gain_rj:.3f} points")

# Local within-state impact for Rio de Janeiro:
rj_orders = len(df[df['customer_state'] == 'RJ'])
rj_low_total = (df[df['customer_state'] == 'RJ']['low_review'] == 1).sum()
pct_drop_within_rj = low_avoided_rj / rj_low_total
score_gain_within_rj = (late_cut_rj * gap_z) / rj_orders
print(f"\n--- LOCAL IMPACT WITHIN RIO DE JANEIRO (12,140 orders) ---")
print(f"RJ Baseline average score: {df[df['customer_state'] == 'RJ']['review_score'].mean():.2f}")
print(f"RJ Low reviews avoided:    {low_avoided_rj:,.0f}")
print(f"RJ Low reviews drop (V%):  {pct_drop_within_rj:.1%}")
print(f"RJ Avg review rise (W):    +{score_gain_within_rj:.2f} stars (from {df[df['customer_state'] == 'RJ']['review_score'].mean():.2f} to {df[df['customer_state'] == 'RJ']['review_score'].mean() + score_gain_within_rj:.2f})")

# Full Marketplace 1/3 Cut:
late_cut_all = n_late * (1/3)
low_avoided_all = late_cut_all * (rate_low_late - rate_low_ontime)
pct_low_drop_all = low_avoided_all / n_low
score_gain_all = (late_cut_all * gap_z) / n_total
print(f"\n--- FULL MARKETPLACE 1/3 LATENESS REDUCTION ---")
print(f"Marketplace late orders eliminated: {late_cut_all:,.0f}")
print(f"Marketplace low reviews avoided:    {low_avoided_all:,.0f}")
print(f"Marketplace low reviews drop (V%):  {pct_low_drop_all:.1%}")
print(f"Marketplace avg review rise (W):    +{score_gain_all:.2f} stars")
