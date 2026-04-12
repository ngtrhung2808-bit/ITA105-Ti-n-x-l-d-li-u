# 3. Xử lý Outliers (Dùng IQR cho cột Price)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping: Giới hạn giá trị trong khoảng an toàn
df['price_capped'] = np.where(df['price'] > upper_bound, upper_bound, 
                             np.where(df['price'] < lower_bound, lower_bound, df['price']))

# 4. Chuẩn hóa số & Biến đổi Categorical
# Scaling diện tích (Min-Max)
scaler = MinMaxScaler()
df['area_scaled'] = scaler.fit_transform(df[['area']])

# One-hot encoding cho District
df = pd.get_dummies(df, columns=['district'], prefix='dist')

# 5. Phát hiện trùng lặp dựa trên Text Similarity (TF-IDF)
tfidf = TfidfVectorizer(stop_words=None)
tfidf_matrix = tfidf.fit_transform(df['description'])

# Tính Cosine Similarity giữa các mô tả
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Tìm các cặp có độ tương đồng > 0.6 (ngưỡng giả định)
duplicates = []
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > 0.6:
            duplicates.append((df.iloc[i]['id'], df.iloc[j]['id'], cosine_sim[i, j]))

print("--- KẾT QUẢ PHÂN TÍCH ---")
print(f"Các cặp bản ghi nghi ngờ trùng lặp: {duplicates}")
print("\nBảng dữ liệu sau khi làm sạch sơ bộ:")
print(df[['id', 'price_capped', 'area_scaled', 'description']].head())
# 5. TRỰC QUAN HÓA (Để xem trong VS Code Interactive hoặc save file)
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['price_capped'])
plt.title("Phân phối giá sau khi xử lý Outlier (Capping)")
plt.show()
