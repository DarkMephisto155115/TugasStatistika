import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

esport = pd.read_csv('ESport_Earnings.csv', on_bad_lines="skip",
                     engine="python", sep=',', encoding='latin-1')

''''
df = esport.select_dtypes(include='number')
print(df.head().to_string())
print("\nData normalisasi dengan z-score\n")
df = esport.select_dtypes(include='number').apply(stats.zscore)
print(df.head().to_string())

df = esport.select_dtypes(include='number')
print("\nData sebelum dinormalisasi dengan cara min-max\n")
print(df.to_string())
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print("\nData setelah dinormalisasi dengan cara min-max\n")
print(normalized_df.to_string())


t10_esports_tp = esport[['GameName','PlayerNo']].set_index('GameName').sort_values('PlayerNo', ascending=False)
t10_esports_te = esport[['GameName','TotalMoney']].set_index('GameName').sort_values('TotalMoney', ascending=False)
t10_esports_pg = esport[['Genre','PlayerNo']].set_index('Genre').groupby('Genre').sum().sort_values('PlayerNo', ascending=False)
t10_esports_eg = esport[['Genre','TotalMoney']].set_index('Genre').groupby('Genre').sum().sort_values('TotalMoney', ascending=False)

t10_esports_tp.head(10).plot(kind='barh')
plt.title('Top 10 Games by Professional Player Count All Time');

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(t10_esports_tp)
normalized_df = pd.DataFrame(normalized_data, columns=t10_esports_tp.columns) 
normalized_df.head(10).plot(kind='barh')
plt.title('Top 10 Games by Professional Player Count All Time');

plt.show()
'''''
# Generate some random data for demonstration
data = df = esport['IdNo']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density distribution
sns.kdeplot(data, fill=True, ax=axes[0])
axes[0].set_title('Original Data')

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.array.reshape(-1, 1)).flatten()

sns.kdeplot(normalized_data, fill=True, ax=axes[1])
axes[1].set_title('Normalized Data')

plt.show()
# Generate so

# Generate some random data for demonstration
data = df = esport['TotalMoney']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density distribution
sns.kdeplot(data, fill=True, ax=axes[0])
axes[0].set_title('Original Data')
axes[0].set
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.array.reshape(-1, 1)).flatten()

sns.kdeplot(normalized_data, fill=True, ax=axes[1])
axes[1].set_title('Normalized Data')

plt.show()
# Generate some random data for demonstration
data = df = esport['PlayerNo']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density distribution
sns.kdeplot(data, fill=True, ax=axes[0])
axes[0].set_title('Original Data')

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.array.reshape(-1, 1)).flatten()

sns.kdeplot(normalized_data, fill=True, ax=axes[1])
axes[1].set_title('Normalized Data')

plt.show()

# Generate some random data for demonstration
data = df = esport['TournamentNo']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density distribution
sns.kdeplot(data, fill=True, ax=axes[0])
axes[0].set_title('Original Data')

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.array.reshape(-1, 1)).flatten()

sns.kdeplot(normalized_data, fill=True, ax=axes[1])
axes[1].set_title('Normalized Data')

plt.show()

# Generate some random data for demonstration
data = df = esport['Top_Country_Earnings']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density distribution
sns.kdeplot(data, fill=True, ax=axes[0])
axes[0].set_title('Original Data')

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.array.reshape(-1, 1)).flatten()

sns.kdeplot(normalized_data, fill=True, ax=axes[1])
axes[1].set_title('Normalized Data')

plt.show()

# Generate some random data for demonstration
data = df = esport['Releaseyear']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density distribution
sns.kdeplot(data, fill=True, ax=axes[0])
axes[0].set_title('Original Data')

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.array.reshape(-1, 1)).flatten()

sns.kdeplot(normalized_data, fill=True, ax=axes[1])
axes[1].set_title('Normalized Data')

plt.show()


'''''
X_minmax = scaler.fit_transform(esport.select_dtypes(include='number'))
print(X_minmax)

scaler = MinMaxScaler()
x = esport.drop(esport.columns[[2,3,6]], axis=1)
X_minmax = scaler.fit_transform(x)
print(X_minmax)
'''''
