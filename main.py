import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
'''''

t10_esports_tp = esport[['GameName','PlayerNo']].set_index('GameName').sort_values('PlayerNo', ascending=False)
t10_esports_te = esport[['GameName','TotalMoney']].set_index('GameName').sort_values('TotalMoney', ascending=False)
t10_esports_pg = esport[['Genre','PlayerNo']].set_index('Genre').groupby('Genre').sum().sort_values('PlayerNo', ascending=False)
t10_esports_eg = esport[['Genre','TotalMoney']].set_index('Genre').groupby('Genre').sum().sort_values('TotalMoney', ascending=False)
t10_esports_tp.head(10).plot(kind='barh')
plt.title('Top 10 Games by Professional Player Count All Time');
plt.show()

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(t10_esports_tp)
normalized_df = pd.DataFrame(normalized_data, columns=t10_esports_tp.columns)
normalized_df.head(10).plot(kind='barh')
plt.title('Top 10 Games by Professional Player Count All Time');
plt.show()