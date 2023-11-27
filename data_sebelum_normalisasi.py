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

# Buat skewness plot untuk setiap kolom numerik sebelum normalisasi
plt.figure(figsize=(16, 6))
plt.subplot(121)

for column in numeric_columns:
    sns.histplot(data[column], label=column, kde=True)

plt.title('Skewness Plot Sebelum Normalisasi')
plt.xlabel('Nilai Asli')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()