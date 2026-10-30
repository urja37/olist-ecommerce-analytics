import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect("olist.duckdb")
df_raw = con.execute("SELECT * FROM order_features").df()

print(f"Initial raw rows: {len(df_raw):,}")

# 1. Audit missing key fields
missing_dates = df_raw[['order_purchase_timestamp', 'order_approved_at', 
                        'order_delivered_carrier_date', 'order_delivered_customer_date', 
                        'order_estimated_delivery_date']].isnull().any(axis=1)

missing_delivered = df_raw['order_delivered_customer_date'].isnull()
missing_carrier = df_raw['order_delivered_carrier_date'].isnull()
missing_approved = df_raw['order_approved_at'].isnull()
missing_distance = df_raw['distance_km'].isnull()
missing_weight = df_raw['total_weight_g'].isnull()

print("\n--- Missing Value Breakdown ---")
print(f"Missing order_delivered_customer_date: {missing_delivered.sum():,}")
print(f"Missing order_delivered_carrier_date:  {missing_carrier.sum():,}")
print(f"Missing order_approved_at:             {missing_approved.sum():,}")
print(f"Missing distance_km:                   {missing_distance.sum():,}")
print(f"Missing total_weight_g:                {missing_weight.sum():,}")

# 2. Audit impossible values
# Ensure timestamps are parsed
for col in ['order_purchase_timestamp', 'order_approved_at', 
            'order_delivered_carrier_date', 'order_delivered_customer_date', 
            'order_estimated_delivery_date']:
    df_raw[col] = pd.to_datetime(df_raw[col])

# Delivery earlier than purchase
neg_actual_delivery = df_raw['actual_days'] < 0

# Carrier pickup after delivery (carrier delivered customer before carrier pickup)
carrier_transit_days = (df_raw['order_delivered_customer_date'] - df_raw['order_delivered_carrier_date']).dt.total_seconds() / 86400.0
neg_transit = carrier_transit_days < 0

# Carrier pickup before purchase timestamp
neg_handling = df_raw['seller_handling_days'] < 0

# Weight <= 0
weight_zero = df_raw['total_weight_g'] <= 0

print("\n--- Impossible Value Breakdown ---")
print(f"Negative actual_days (delivery < purchase):          {neg_actual_delivery.sum():,}")
print(f"Negative transit days (delivery < carrier pickup):   {neg_transit.sum():,}")
print(f"Negative handling days (carrier pickup < purchase):  {neg_handling.sum():,}")
print(f"Weight <= 0 grams:                                   {weight_zero.sum():,}")

# 3. Clean step-by-step and track drop counts
df = df_raw.copy()
drop_log = []

# Step A: Drop missing delivery date (vital for delivery analysis)
mask_drop_deliv = df['order_delivered_customer_date'].isnull()
count_a = mask_drop_deliv.sum()
df = df[~mask_drop_deliv].copy()
drop_log.append(("Missing order_delivered_customer_date", count_a, len(df)))

# Step B: Drop missing carrier pickup or approval timestamps
mask_drop_other_dates = df['order_delivered_carrier_date'].isnull() | df['order_approved_at'].isnull()
count_b = mask_drop_other_dates.sum()
df = df[~mask_drop_other_dates].copy()
drop_log.append(("Missing carrier date or approval timestamp", count_b, len(df)))

# Step C: Drop missing distance_km (missing coordinates for customer or seller)
mask_drop_dist = df['distance_km'].isnull()
count_c = mask_drop_dist.sum()
df = df[~mask_drop_dist].copy()
drop_log.append(("Missing geolocation distance (unmapped zip code)", count_c, len(df)))

# Step D: Drop missing total_weight_g
mask_drop_weight = df['total_weight_g'].isnull()
count_d = mask_drop_weight.sum()
df = df[~mask_drop_weight].copy()
drop_log.append(("Missing product weight", count_d, len(df)))

# Step E: Drop impossible values: weight <= 0
mask_weight_zero = df['total_weight_g'] <= 0
count_e = mask_weight_zero.sum()
df = df[~mask_weight_zero].copy()
drop_log.append(("Impossible product weight (<= 0 g)", count_e, len(df)))

# Step F: Drop impossible delivery times (actual_days < 0 or carrier_transit < 0 or handling < 0)
transit_days = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.total_seconds() / 86400.0
mask_impossible_timing = (df['actual_days'] < 0) | (transit_days < 0) | (df['seller_handling_days'] < 0)
count_f = mask_impossible_timing.sum()
df = df[~mask_impossible_timing].copy()
drop_log.append(("Impossible timestamp chronology (negative transit or handling)", count_f, len(df)))

print("\n=== STEP-BY-STEP FILTERING LOG ===")
log_df = pd.DataFrame(drop_log, columns=["Reason for Drop", "Rows Removed", "Remaining Rows"])
print(log_df.to_string(index=False))

total_removed = len(df_raw) - len(df)
pct_retained = len(df) * 100.0 / len(df_raw)
print(f"\nTotal rows removed: {total_removed:,} ({100 - pct_retained:.2f}%)")
print(f"Final clean dataset rows: {len(df):,} ({pct_retained:.2f}% retained)")

# 4. Outlier capping (Winsorization) & Log transforms
p99_weight = df['total_weight_g'].quantile(0.99)
p99_freight_ratio = df['freight_ratio'].quantile(0.99)
p99_volume = df['total_volume_cm3'].quantile(0.99)

print(f"\n--- 99th Percentile Thresholds ---")
print(f"Total Weight 99th percentile:         {p99_weight:.1f} g (Max raw: {df['total_weight_g'].max():.1f} g)")
print(f"Freight Ratio 99th percentile:        {p99_freight_ratio:.4f} (Max raw: {df['freight_ratio'].max():.4f})")
print(f"Total Volume 99th percentile:         {p99_volume:.1f} cm3 (Max raw: {df['total_volume_cm3'].max():.1f} cm3)")

# Add capped columns
df['weight_capped'] = df['total_weight_g'].clip(upper=p99_weight)
df['freight_ratio_capped'] = df['freight_ratio'].clip(upper=p99_freight_ratio)
df['volume_capped'] = df['total_volume_cm3'].clip(upper=p99_volume)

# Add log1p transforms for skewed variables
df['log_weight'] = np.log1p(df['total_weight_g'])
df['log_volume'] = np.log1p(df['total_volume_cm3'])
df['log_price'] = np.log1p(df['total_price'])
df['log_freight'] = np.log1p(df['total_freight'])
df['log_distance'] = np.log1p(df['distance_km'])

# Save cleaned table back into DuckDB as 'order_master' or 'order_features_clean'
con.register("df_cleaned_view", df)
con.execute("CREATE OR REPLACE TABLE order_master AS SELECT * FROM df_cleaned_view")
print("\nSuccessfully created and saved clean table 'order_master' in olist.duckdb!")
