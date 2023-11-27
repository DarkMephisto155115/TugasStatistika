import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Baca data dari file CSV
file_path = "Car_sales.csv"
data = pd.read_csv(file_path)

# Pilih kolom numerik untuk dianalisis
numeric_columns = data.select_dtypes(include=[float, int]).columns
data.drop()
# Buat skewness plot untuk setiap kolom numerik sebelum normalisasi
for column in numeric_columns:
    # Hitung skewness
    skewness_value = data[column].skew()

    # Buat skewness plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True, color='blue', edgecolor='black')
    plt.annotate(f'Skewness: {skewness_value:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
    plt.title(f'Skewness Plot - {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Tampilkan legenda
    plt.legend()

    # Transformasi Min-Max Scaling
    min_max_scaler = MinMaxScaler()
    data_min_max = min_max_scaler.fit_transform(data[[column]])

    # Buat histogram untuk data setelah Min-Max Scaling
    plt.subplot(1, 2, 2)
    sns.histplot(data_min_max, kde=True, color='green', edgecolor='black')
    plt.title(f'Min-Max Scaling - {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Tampilkan plot
    plt.tight_layout()
    plt.show()

    # Transformasi Z-score Scaling
    zscore_scaler = StandardScaler()
    data_zscore = zscore_scaler.fit_transform(data[[column]])

    # Buat histogram untuk data setelah Z-score Scaling
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data_zscore, kde=True, color='orange', edgecolor='black')
    plt.title(f'Z-score Scaling - {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Tampilkan plot
    plt.tight_layout()
    plt.show()