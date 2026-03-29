import sys
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập font để hiển thị tiếng Việt (nếu cần trên một số hệ thống)
# plt.rcParams['font.family'] = 'Arial' 

# ---------------------------------------------------------
# 1. TẠO DỮ LIỆU MẪU (Dữ liệu "bẩn" để thực hành)
# ---------------------------------------------------------

def generate_sample_data():
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'id': range(1, n_rows + 1),
        'gia_nha': np.random.normal(3000, 1000, n_rows).tolist(), # Giá nhà (triệu VNĐ)
        'dien_tich': np.random.normal(70, 20, n_rows).tolist(),    # Diện tích (m2)
        'so_phong': np.random.randint(1, 6, n_rows).tolist(),      # Số phòng
        'tinh_trang': np.random.choice(['Mới', 'Cũ', 'Sửa sang', 'Mơi', None], n_rows), # Lỗi typo 'Mơi' và missing
        'vi_tri': np.random.choice(['Quận 1', 'Quận 7', 'Quận Thủ Đức', 'Bình Thạnh', None], n_rows),
        'ngay_giao_dich': pd.date_range(start='2023-01-01', periods=n_rows, freq='D').tolist()
    }
    
    df = pd.DataFrame(data)
    
    # Tạo lỗi dữ liệu để thực hành làm sạch
    # 1. Giá trị thiếu (Missing values)
    df.loc[10:15, 'gia_nha'] = np.nan
    df.loc[20:22, 'dien_tich'] = np.nan
    
    # 2. Giá trị không hợp lệ (Invalid values)
    df.loc[5, 'gia_nha'] = -500  # Giá âm
    df.loc[8, 'so_phong'] = 0    # Số phòng bằng 0
    
    # 3. Dữ liệu trùng lặp (Duplicates)
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    
    return df

# Khởi tạo DataFrame
df = generate_sample_data()

print("--- DỮ LIỆU BAN ĐẦU ---")
print(df.head())
print("\n")

# ---------------------------------------------------------
# GIAI ĐOẠN 1: KHÁM PHÁ DỮ LIỆU ĐA DẠNG (EDA)
# ---------------------------------------------------------

print("--- 1.1 PHÂN TÍCH THỐNG KÊ ---")
# Tính toán mean, median, std, min, max
stats = df.describe()
print(stats)

# Kiểm tra dữ liệu thiếu
missing_values = df.isnull().sum()
print("\nSố lượng giá trị thiếu mỗi cột:")
print(missing_values)

# Kiểm tra dữ liệu trùng lặp
duplicates = df.duplicated().sum()
print(f"\nSố lượng dòng trùng lặp: {duplicates}")

print("\n--- 1.2 VẼ BIỂU ĐỒ (VISUALIZATION) ---")
# Tạo canvas cho các biểu đồ
plt.figure(figsize=(15, 10))

# Histogram cho Giá nhà
plt.subplot(2, 2, 1)
sns.histplot(df['gia_nha'].dropna(), kde=True, color='blue')
plt.title('Phân phối Giá nhà (Histogram)')

# Boxplot cho Diện tích
plt.subplot(2, 2, 2)
sns.boxplot(x=df['dien_tich'], color='green')
plt.title('Biểu đồ Boxplot Diện tích')

# Violin plot cho Số phòng và Giá nhà
plt.subplot(2, 2, 3)
sns.violinplot(x='so_phong', y='gia_nha', data=df)
plt.title('Violin Plot: Số phòng vs Giá nhà')

# Phân phối biến Categorical (Vị trí)
plt.subplot(2, 2, 4)
sns.countplot(y='vi_tri', data=df, order=df['vi_tri'].value_counts().index)
plt.title('Phân phối theo Vị trí')

plt.tight_layout()
plt.savefig('eda_plots.png') # Lưu biểu đồ ra file ảnh
print("Đã lưu biểu đồ EDA vào file 'eda_plots.png'")

# ---------------------------------------------------------
# GIAI ĐOẠN 2: XỬ LÝ DỮ LIỆU BẨN (CLEANING)
# ---------------------------------------------------------

print("\n--- 2.1 XỬ LÝ GIÁ TRỊ THIẾU (MISSING VALUES) ---")

# Điền missing cho gia_nha bằng Median (vì giá thường có outlier)
df['gia_nha'] = df['gia_nha'].fillna(df['gia_nha'].median())

# Điền missing cho dien_tich bằng Mean
df['dien_tich'] = df['dien_tich'].fillna(df['dien_tich'].mean())

# Điền missing cho tinh_trang bằng Mode (Giá trị xuất hiện nhiều nhất)
df['tinh_trang'] = df['tinh_trang'].fillna(df['tinh_trang'].mode()[0])

print("Đã xử lý xong các giá trị thiếu.")

print("\n--- 2.2 XỬ LÝ DỮ LIỆU KHÔNG HỢP LỆ ---")

# Loại bỏ giá trị âm (hoặc chuyển thành trị tuyệt đối nếu logic cho phép)
df = df[df['gia_nha'] > 0]

# Xử lý số phòng = 0 (có thể gán bằng 1 hoặc median)
df.loc[df['so_phong'] == 0, 'so_phong'] = 1

# Sửa lỗi typo trong categorical (Mơi -> Mới)
df['tinh_trang'] = df['tinh_trang'].replace('Mơi', 'Mới')

# Loại bỏ dòng trùng lặp
df = df.drop_duplicates()

print(f"Kích thước tập dữ liệu sau khi làm sạch: {df.shape}")
print("\n--- DỮ LIỆU SAU KHI LÀM SẠCH ---")
print(df.head())

# --- 2.3 FEATURE ENGINEERING (TẠO BIẾN MỚI) ---
# Tính giá mỗi mét vuông để so sánh hiệu quả hơn
df['gia_theo_m2'] = df['gia_nha'] / df['dien_tich']
print(f"\nĐã tạo biến 'gia_theo_m2'. Giá trung bình: {df['gia_theo_m2'].mean():.2f} triệu/m2")

# Lưu kết quả làm sạch
df.to_csv('cleaned_data.csv', index=False)
print("\nĐã lưu dữ liệu sạch vào file 'cleaned_data.csv'")

# ---------------------------------------------------------
# GIAI ĐOẠN 3: BIẾN ĐỔI DỮ LIỆU (TRANSFORMATION)
# ---------------------------------------------------------

print("\n--- 3.1 CHUẨN HÓA DỮ LIỆU (MIN-MAX SCALING) ---")
# Đưa giá trị về khoảng [0, 1] để các thuật toán machine learning chạy tốt hơn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols_to_scale = ['gia_nha', 'dien_tich', 'gia_theo_m2']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(f"Đã chuẩn hóa Min-Max cho các cột: {cols_to_scale}")

print("\n--- 3.3 XỬ LÝ OUTLIER (IQR METHOD) ---")
# Loại bỏ các giá trị ngoại lai (quá cao hoặc quá thấp)
Q1 = df['gia_nha'].quantile(0.25)
Q3 = df['gia_nha'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[(df['gia_nha'] >= lower_bound) & (df['gia_nha'] <= upper_bound)]
print(f"Số lượng dòng sau khi loại bỏ Outlier: {len(df_no_outliers)} (Bỏ {len(df) - len(df_no_outliers)} dòng)")
df = df_no_outliers

print("\n--- 3.4 PHÂN NHÓM DỮ LIỆU (BINNING) ---")
# Chia diện tích thành 3 loại: Nhỏ, Trung bình, Lớn
bins = [0, 0.3, 0.7, 1.1] 
labels = ['Nhỏ', 'Trung bình', 'Lớn']
df['nhom_dien_tich'] = pd.cut(df['dien_tich'], bins=bins, labels=labels)

print("Phân bố theo nhóm diện tích:")
print(df['nhom_dien_tich'].value_counts())

print("\n--- 3.5 MÃ HÓA BIẾN CATEGORICAL (ONE-HOT ENCODING) ---")
# Chuyển đổi 'tinh_trang' sang các cột số
df_dummies = pd.get_dummies(df['tinh_trang'], prefix='tinh_trang')
# Chỉ thêm các cột chưa tồn tại
new_cols = [c for c in df_dummies.columns if c not in df.columns]
df = pd.concat([df, df_dummies[new_cols]], axis=1)

print(f"Đã thêm các cột One-Hot: {new_cols}")

# ---------------------------------------------------------
# GIAI ĐOẠN 4: PHÂN TÍCH SÂU VÀ TRỰC QUAN HÓA THÊM
# ---------------------------------------------------------

print("\n--- 4.1 PHÂN TÍCH BIẾN SỐ THEO NHÓM (GROUPBY) ---")
# Tính giá nhà trung bình theo từng Vị trí
avg_price_by_location = df.groupby('vi_tri')['gia_nha'].mean().sort_values(ascending=False)
print("Giá nhà trung bình (đã chuẩn hóa) theo Vị trí:")
print(avg_price_by_location)

print("\n--- 4.2 MA TRẬN TƯƠNG QUAN (CORRELATION MATRIX) ---")
# Tự động chọn các cột kiểu số để tính tương quan
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận tương quan giữa các biến số')
plt.savefig('correlation_heatmap.png')
print("Đã lưu biểu đồ Heatmap vào file 'correlation_heatmap.png'")

# ---------------------------------------------------------
# HOÀN TẤT VÀ LƯU KẾT QUẢ CUỐI CÙNG
# ---------------------------------------------------------

print("\n--- TỔNG KẾT ---")
print(f"Tổng số dòng dữ liệu: {len(df)}")
print(f"Danh sách các cột hiện có: {list(df.columns)}")

# Lưu kết quả cuối cùng ra file CSV
df.to_csv('processed_data_final.csv', index=False)
print("\nĐã lưu dữ liệu sau khi biến đổi đầy đủ vào file 'processed_data_final.csv'")
print("--- KẾT THÚC QUY TRÌNH TIỀN XỬ LÝ ---")
