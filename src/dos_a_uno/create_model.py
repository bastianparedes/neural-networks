import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from dos_a_uno.create_data import create_data

# =========================
# 1. DATA LOADING FROM CSV
# =========================

create_data()

try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found.")
    exit()

# Features y target
X = df[['x1', 'x2']].values
Y = df['y1'].values.reshape(-1, 1)

# =========================
# 2. DATA NORMALIZATION
# =========================

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(), Y.std()

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# 🔥 GUARDAR SCALER (CLAVE)
with open('scaler.pkl', 'wb') as f:
    pickle.dump((X_mean, X_std, Y_mean, Y_std), f)

# =========================
# 3. MODEL DEFINITION
# =========================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, activation='linear', input_shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(optimizer='adam', loss='mse')

# =========================
# 4. TRAINING
# =========================

print('Training the model with x1 and x2...')
model.fit(X_norm, Y_norm, epochs=200, verbose=0)

# =========================
# 5. PREDICTION (Visualization Grid)
# =========================

x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)

X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

X_test = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]
X_test_norm = (X_test - X_mean) / X_std

Y_pred_norm = model.predict(X_test_norm, verbose=0)
Y_pred = Y_pred_norm * Y_std + Y_mean
Y_pred_mesh = Y_pred.reshape(X1_mesh.shape)

# 🔥 GUARDAR MODELO
model.save('model.keras')

# =========================
# 6. VISUALIZATION (3D)
# =========================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Datos reales
ax.scatter(
    X[:, 0],
    X[:, 1],
    Y.flatten(),
    color='red',
    label='Actual Data',
    alpha=1,
)

# Superficie del modelo
ax.plot_surface(
    X1_mesh,
    X2_mesh,
    Y_pred_mesh,
    color='blue',
    alpha=0.4,
)

ax.set_title('Neural Network Fit: Y1 = f(X1, X2)')
ax.set_xlabel('X1 Axis')
ax.set_ylabel('X2 Axis')
ax.set_zlabel('Y1 Axis')

plt.show()
