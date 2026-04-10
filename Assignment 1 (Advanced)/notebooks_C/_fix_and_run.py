#!/usr/bin/env python3
"""Fix task1c notebook for wide-format cleaned data and execute it."""
import json
import copy

NB_PATH = "task1c_feature_engineering.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

def find_cell(cell_id):
    for i, c in enumerate(nb['cells']):
        if c.get('id') == cell_id:
            return i
    return None

def set_code(cell_id, new_code):
    idx = find_cell(cell_id)
    if idx is None:
        raise ValueError(f"Cell {cell_id} not found")
    nb['cells'][idx]['source'] = new_code.split('\n')
    # Fix: each line except last needs trailing \n
    lines = new_code.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    else:
        result.append('')
    nb['cells'][idx]['source'] = result
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

def set_md(cell_id, new_md):
    idx = find_cell(cell_id)
    if idx is None:
        raise ValueError(f"Cell {cell_id} not found")
    lines = new_md.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    else:
        result.append('')
    nb['cells'][idx]['source'] = result

# --- Fix cell e5ef2474: Load data (handle wide format) ---
set_code('e5ef2474', '''import os

cleaned_path = '../data/dataset_mood_smartphone_cleaned.csv'
raw_path = '../data/dataset_mood_smartphone.csv'

if os.path.exists(cleaned_path):
    df_wide = pd.read_csv(cleaned_path)
    df_wide['date'] = pd.to_datetime(df_wide['date']).dt.date
    data_source = "cleaned (wide format)"
    print(f"Loaded cleaned dataset from {cleaned_path}")
else:
    print(f"Cleaned dataset not found at {cleaned_path}. Loading raw data as fallback...")
    df_raw = pd.read_csv(raw_path)
    if df_raw.columns[0] == '' or df_raw.columns[0].startswith('Unnamed'):
        df_raw = df_raw.drop(columns=df_raw.columns[0])
    df_raw['time'] = pd.to_datetime(df_raw['time'])
    df_raw['date'] = df_raw['time'].dt.date
    df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')
    # We will handle aggregation below
    df_wide = None
    data_source = "raw (long format)"
    print(f"Loaded raw dataset from {raw_path}")

if df_wide is not None:
    feature_vars_all = [c for c in df_wide.columns if c not in ['id', 'date']]
    print(f"\\nData source: {data_source}")
    print(f"Dataset shape: {df_wide.shape}")
    print(f"Number of patients: {df_wide['id'].nunique()}")
    print(f"Variables: {sorted(feature_vars_all)}")
    print(f"Date range: {min(df_wide['date'])} to {max(df_wide['date'])}")
    df_wide.head()''')

# --- Fix cell 16e20af7: Aggregation vars (adapt for wide format) ---
set_code('16e20af7', '''# The cleaned data is already in wide format (one row per patient per day).
# We identify variable types for documentation and NaN-filling purposes.
mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
sum_vars = ['screen'] + [c for c in df_wide.columns if c.startswith('appCat.')]
count_vars = ['call', 'sms']

print("Variables aggregated by MEAN (in raw data):", mean_vars)
print("Variables aggregated by SUM (in raw data):", sum_vars)
print("Variables aggregated by COUNT (in raw data):", count_vars)''')

# --- Fix cell 811e502f: Skip aggregation, use wide data directly ---
set_code('811e502f', '''# The cleaned dataset is already aggregated into daily wide format.
# We use it directly as our daily_df.

# def aggregate_daily(...):  # Not needed - data is already aggregated
#     ...

daily_df = df_wide.copy()

print(f"Daily data shape: {daily_df.shape}")
print(f"Patients: {daily_df['id'].nunique()}, Days per patient (approx): {daily_df.groupby('id')['date'].count().mean():.0f}")
print(f"\\nColumns: {list(daily_df.columns)}")
daily_df.head(10)''')

# --- Fix cell 9b011961: Missing value handling ---
set_code('9b011961', '''# Check missing values in the daily data
missing = daily_df.isnull().sum()
missing_pct = (daily_df.isnull().sum() / len(daily_df) * 100).round(1)
missing_info = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
print("Missing values in daily data:")
print(missing_info[missing_info['missing_count'] > 0])

# Fill missing daily values with 0 for count/sum variables (no observation = no activity)
fill_zero_cols = [c for c in sum_vars + count_vars if c in daily_df.columns]
daily_df[fill_zero_cols] = daily_df[fill_zero_cols].fillna(0)

print(f"\\nAfter filling count/sum NaNs with 0: {daily_df.isnull().sum().sum()} remaining NaNs")
print("Remaining NaN columns:", list(daily_df.columns[daily_df.isnull().any()]))''')

# --- Fix cell c3051e13: Fix closing paren typo ---
set_code('c3051e13', '''# Apply sliding window feature engineering per patient
patient_ids = sorted(daily_df['id'].unique())
all_instances = []

for pid in patient_ids:
    patient_data = daily_df[daily_df['id'] == pid].copy()
    patient_features = build_sliding_window_features(
        patient_data, feature_vars, WINDOW_SIZE, N_LAGS
    )
    all_instances.append(patient_features)

features_df = pd.concat(all_instances, ignore_index=True)

print(f"Total instances created: {len(features_df)}")
print(f"Patients represented: {features_df['id'].nunique()}")
print(f"Instances per patient:")
print(features_df.groupby('id').size().describe())
print(f"\\nTotal features: {len([c for c in features_df.columns if c not in ['id', 'date', 'target_mood']])}")''')

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
