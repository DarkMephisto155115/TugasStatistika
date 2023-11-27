import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Baca data dari file CSV
file_path = r"Car_sales.csv"
data = pd.read_csv(file_path)

# Hapus kolom 'ORDERDATE' dari DataFrame
data = data.drop('ORDERDATE', axis=1, errors='ignore')

# Pilih kolom numerik untuk normalisasi
numeric_columns = data.select_dtypes(include=[float, int]).columns

# Normalisasi menggunakan Min-Max Scaling
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

# Menambahkan kolom "STATUS dan PRODUCTLINE" pada DataFrame yang sudah dinormalisasi
data_normalized[['STATUS', 'PRODUCTLINE']] = data[['STATUS', 'PRODUCTLINE']]

# Menampilkan data setelah normalisasi dengan kolom "ORDERDATE" di depan
columns_order = ['ORDERDATE'] + [col for col in data_normalized.columns if col not in ['STATUS', 'PRODUCTLINE']]
data_normalized = data_normalized[columns_order[1:]]

# Tampilkan data setelah normalisasi
print("Data setelah normalisasi:")
print(data_normalized)

# Buat skewness plot untuk setiap kolom numerik pada DataFrame yang sudah dinormalisasi
plt.figure(figsize=(16, 6))
plt.subplot(121)

for column in numeric_columns:
    # Skip kolom 'STATUS dan PRODUCTLINE' saat membuat plot
    if column != 'STATUS, PRODUCTLINE':
        sns.histplot(data_normalized[column], label=column, kde=True)

plt.title('Skewness Plot Setelah Normalisasi')
plt.xlabel('Nilai Normalisasi')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()