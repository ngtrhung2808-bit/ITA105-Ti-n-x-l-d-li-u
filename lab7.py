import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
df = pd.read_csv('ITA105_Lab_7.csv')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# --- BÀI 1: Phân tích skewness ---
skewness = df[numeric_cols].skew().sort_values(ascending=False)
top_10_skewed = skewness.head(10)
print('Top 10 skewed columns:')
print(top_10_skewed)

top_3_cols = skewness.abs().sort_values(ascending=False).index[:3].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(top_3_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='royalblue')
    axes[i].set_title(f'Distribution of {col}\nSkew: {df[col].skew():.2f}')
plt.tight_layout()
plt.savefig('plots/bai1_distributions.png')
plt.close()

# --- BÀI 2: Biến đổi dữ liệu ---
# Choose 2 positive cols: SalePrice, LotArea
# Choose 1 col with neg/0: NegSkewIncome (checked from preview) or MixedFeature
pos_cols = ['SalePrice', 'LotArea']
neg_col = 'NegSkewIncome'

results_b2 = []

# Transforms
for col in pos_cols:
    # Original
    skew_orig = df[col].skew()
    
    # Log
    df_log = np.log1p(df[col])
    skew_log = df_log.skew()
    
    # Box-Cox
    df_bc, lam = stats.boxcox(df[col] + 1)
    skew_bc = pd.Series(df_bc).skew()
    
    # Power (Yeo-Johnson)
    pt = PowerTransformer(method='yeo-johnson')
    df_pt = pt.fit_transform(df[[col]])
    skew_pt = pd.Series(df_pt.flatten()).skew()
    
    results_b2.append({
        'Column': col,
        'Original': skew_orig,
        'Log': skew_log,
        'Box-Cox': skew_bc,
        'Yeo-Johnson': skew_pt
    })

# Neg col
skew_orig_n = df[neg_col].skew()
pt_n = PowerTransformer(method='yeo-johnson')
df_pt_n = pt_n.fit_transform(df[[neg_col]])
skew_pt_n = pd.Series(df_pt_n.flatten()).skew()

results_b2.append({
    'Column': neg_col,
    'Original': skew_orig_n,
    'Log': np.nan,
    'Box-Cox': np.nan,
    'Yeo-Johnson': skew_pt_n
})

results_df = pd.DataFrame(results_b2)
print('\nSkewness Comparison Table:')
print(results_df)

# Plot Before vs After for SalePrice (Yeo-Johnson)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='teal')
axes[0].set_title('SalePrice Original')
sns.histplot(df_pt.flatten(), kde=True, ax=axes[1], color='orange')
axes[1].set_title('SalePrice Yeo-Johnson Transformed')
plt.savefig('plots/bai2_transform_compare.png')
plt.close()

# --- BÀI 3: Mô hình hóa ---
# Features: all numeric except SalePrice
X = df[numeric_cols].drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Version A: Raw
model_a = LinearRegression()
model_a.fit(X_train, y_train)
y_pred_a = model_a.predict(X_test)
rmse_a = np.sqrt(mean_squared_error(y_test, y_pred_a))
r2_a = r2_score(y_test, y_pred_a)

# Version B: Log target
model_b = LinearRegression()
model_b.fit(X_train, np.log1p(y_train))
y_pred_b_log = model_b.predict(X_test)
y_pred_b = np.expm1(y_pred_b_log)
rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_b))
r2_b = r2_score(y_test, y_pred_b)

# Version C: Transform skewed features (X)
pt_c = PowerTransformer(method='yeo-johnson')
X_train_c = pt_c.fit_transform(X_train)
X_test_c = pt_c.transform(X_test)

model_c = LinearRegression()
model_c.fit(X_train_c, y_train)
y_pred_c = model_c.predict(X_test_c)
rmse_c = np.sqrt(mean_squared_error(y_test, y_pred_c))
r2_c = r2_score(y_test, y_pred_c)

print('\nModel Comparison:')
print(f'Ver A (Raw): RMSE={rmse_a:.2f}, R2={r2_a:.4f}')
print(f'Ver B (Log Target): RMSE={rmse_b:.2f}, R2={r2_b:.4f}')
print(f'Ver C (Transformed Features): RMSE={rmse_c:.2f}, R2={r2_c:.4f}')

# --- BÀI 4: Business Insight ---
# Create log-price-index
df['log-price-index'] = np.log1p(df['SalePrice'])
plt.figure(figsize=(10, 6))
sns.boxplot(x='Neighborhood', y='log-price-index', data=df)
plt.title('Log Price Index by Neighborhood')
plt.savefig('plots/bai4_business_index.png')
plt.close()

results_df.to_csv('plots/results_table.csv', index=False)
