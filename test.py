# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from distribute_data import *
from confeddi import FederatedSystem
import os


# %%
data = pd.read_csv('RTT_data.csv')

# Getting rid of complex-valued columns
data = data.select_dtypes(exclude = ['object'])

# Quick look at data
print(f'Number of samples: {data.shape[0]}')
print(f'Features per sample: {data.shape[1] - 1}\n')
print(f'Columns:')
for i in data.columns[:-1]:
    if i == 'GroundTruthRange[m]':
        continue
    print(f'{i}, ', end = '')
print(data.columns[-1], end = '')


# Separate data and labels
X = data.drop(columns = ['GroundTruthRange[m]']).to_numpy()
y = data['GroundTruthRange[m]'].to_numpy()
# %%
X
# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 11)

data = generate_data(X_train, y_train, seed = 11)
scaler = StandardScaler()
X_val = scaler.fit_transform(X_val)
X = scaler.fit_transform(X)

# %%
X_train.shape
# %%
client_data = data['Client Data']

# %%
total = X.shape[0]
for client in client_data:
    print(f'Percent of samples: {100 * len(client) / total:.2f}%')
# %%
