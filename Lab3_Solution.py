import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set(style="whitegrid")

# Automatically detect the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = script_dir

files = {
    "Finance": "ITA105_Lab_3_Finance.csv",
    "Gaming": "ITA105_Lab_3_Gaming.csv",
    "Health": "ITA105_Lab_3_Health.csv",
    "Sports": "ITA105_Lab_3_Sports.csv"
}

datasets = {}

print("--- TASK 1: Exploratory Data Analysis (EDA) ---")

for name, filename in files.items():
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    datasets[name] = df
    
    print(f"\n[Dataset: {name}]")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nDescriptive Statistics (Mean, Median, Std):")
    stats = df.agg(['mean', 'median', 'std']).transpose()
    print(stats)

print("\n--- TASK 2: Data Preprocessing ---")

# 2.1 Outlier Detection (using IQR)
def detect_outliers_iqr(df):
    outliers_info = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_info[col] = len(outliers)
    return outliers_info

for name, df in datasets.items():
    print(f"\nOutliers in {name} (IQR method):")
    print(detect_outliers_iqr(df))

# 2.2 Normalization (Min-Max Scaling)
def normalize(df):
    df_numeric = df.select_dtypes(include=[np.number])
    return (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

# 2.3 Standardization (Z-score)
def standardize(df):
    df_numeric = df.select_dtypes(include=[np.number])
    return (df_numeric - df_numeric.mean()) / df_numeric.std()

# Process and Export
for name, df in datasets.items():
    df_norm = normalize(df)
    df_std = standardize(df)
    
    # Save to files
    df_norm.to_csv(os.path.join(data_dir, f"{name}_normalized.csv"), index=False)
    df_std.to_csv(os.path.join(data_dir, f"{name}_standardized.csv"), index=False)
    print(f"Exported normalized and standardized files for {name}.")

print("\n--- TASK 3: Visualization ---")

# 3.1 Histograms for Distribution Comparison (e.g., Finance - doanh_thu_musd)
target_ds = "Finance"
target_col = "doanh_thu_musd"
df = datasets[target_ds]
df_norm = normalize(df)
df_std = standardize(df)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df[target_col], kde=True, color='blue')
plt.title(f'Original {target_col}')

plt.subplot(1, 3, 2)
sns.histplot(df_norm[target_col], kde=True, color='green')
plt.title(f'Normalized {target_col}')

plt.subplot(1, 3, 3)
sns.histplot(df_std[target_col], kde=True, color='red')
plt.title(f'Standardized {target_col}')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "distribution_comparison.png"))
print("Saved distribution comparison plot.")

# 3.2 Scatter Plots
# Finance: Revenue vs Profit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=datasets["Finance"], x="doanh_thu_musd", y="loi_nhuan_musd")
plt.title("Finance: Revenue vs Profit")
plt.savefig(os.path.join(data_dir, "finance_scatter.png"))

# Sports: Height vs Weight
plt.figure(figsize=(8, 6))
sns.scatterplot(data=datasets["Sports"], x="chieu_cao_cm", y="can_nang_kg")
plt.title("Sports: Height vs Weight")
plt.savefig(os.path.join(data_dir, "sports_scatter.png"))
print("Saved scatter plots.")
