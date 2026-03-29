import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tự động lấy đường dẫn đến thư mục chứa file code hiện tại
base_path = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(base_path, filename)

def process_ecommerce():
    print("\n--- Đang xử lý dữ liệu Ecommerce ---")
    # Sử dụng đường dẫn tuyệt đối để tránh lỗi FileNotFoundError
    df = pd.read_csv(get_path('ITA105_Lab_2_Ecommerce.csv'))
    
    # 2. Xử lý cột 'category' bị 'Unknown'
    most_freq_cat = df['category'].mode()[0]
    df['category'] = df['category'].replace('Unknown', most_freq_cat)
    
    # 3. Xử lý cột 'rating' 
    df['rating'] = df['rating'].clip(1, 5)
    
    # 4. Xử lý cột 'price'
    df = df[(df['price'] > 1) & (df['price'] < 1000)]
    
    df.to_csv(get_path('Ecommerce_cleaned.csv'), index=False)
    print("Dữ liệu Ecommerce đã được làm sạch và lưu tại 'Ecommerce_cleaned.csv'")
    return df

def process_housing():
    print("\n--- Đang xử lý dữ liệu Housing ---")
    df = pd.read_csv(get_path('ITA105_Lab_2_Housing.csv'))
    df.columns = ['area', 'price', 'rooms']
    
    # Lọc bỏ giá trị ngoại lệ (chỉ giữ nhà diện tích < 500m2)
    df = df[(df['area'] > 0) & (df['area'] < 500)]
    
    df.to_csv(get_path('Housing_cleaned.csv'), index=False)
    print("Dữ liệu Housing đã được làm sạch và lưu tại 'Housing_cleaned.csv'")
    return df

def process_iot():
    print("\n--- Đang xử lý dữ liệu IoT ---")
    df = pd.read_csv(get_path('ITA105_Lab_2_Iot.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Thay thế nhiệt độ nhiễu (>40 độ) bằng trung bình nhiệt độ của các cảm biến
    mean_temp = df[df['temperature'] <= 40]['temperature'].mean()
    df.loc[df['temperature'] > 40, 'temperature'] = mean_temp
    
    # Độ ẩm giới hạn trong [0, 100]
    df['humidity'] = df['humidity'].clip(0, 100)
    
    df.to_csv(get_path('Iot_cleaned.csv'), index=False)
    print("Dữ liệu IoT đã được làm sạch và lưu tại 'Iot_cleaned.csv'")
    return df

if __name__ == "__main__":
    ecommerce_df = process_ecommerce()
    housing_df = process_housing()
    iot_df = process_iot()
    
    # Tạo biểu đồ phân phối đơn giản
    plt.figure(figsize=(10, 6))
    sns.histplot(ecommerce_df['price'], bins=30, kde=True)
    plt.title('Phân phối giá Ecommerce sau khi làm sạch')
    plt.savefig(get_path('lab2_visualization.png'))
    
    print("\n=== HOÀN THÀNH: Bạn hãy kiểm tra các file '_cleaned.csv' trong thư mục nhé! ===")
    print("Biểu đồ minh họa đã được lưu tại 'lab2_visualization.png'")
