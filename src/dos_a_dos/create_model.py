import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from dos_a_dos.create_data import create_data

# =========================
# 1. DATA LOADING FROM CSV
# =========================

create_data()

try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found.")
    exit()

# Features y target (🔥 ahora 2 outputs)
X = df[['x1', 'x2']].values
Y = df[['y1', 'y2']].values  # <-- CAMBIO CLAVE

# =========================
# 2. DATA NORMALIZATION
# =========================

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)  # <-- vector ahora

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# Guardar scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump((X_mean, X_std, Y_mean, Y_std), f)

# =========================
# 3. MODEL DEFINITION
# =========================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, activation='linear', input_shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2),  # <-- CAMBIO: 2 outputs
    ]
)

model.compile(optimizer='adam', loss='mse')

# =========================
# 4. TRAINING
# =========================

print('Training the model with x1 and x2 predicting y1, y2...')
model.fit(X_norm, Y_norm, epochs=200, verbose=0)

# =========================
# 5. PREDICTION (GRID)
# =========================

x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)

X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

X_test = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]
X_test_norm = (X_test - X_mean) / X_std

Y_pred_norm = model.predict(X_test_norm, verbose=0)
Y_pred = Y_pred_norm * Y_std + Y_mean  # broadcasting correcto

# Separar outputs
Y1_pred = Y_pred[:, 0].reshape(X1_mesh.shape)
Y2_pred = Y_pred[:, 1].reshape(X1_mesh.shape)

# Guardar modelo
model.save('model.keras')

# =========================
# 6. VISUALIZATION (VECTOR FIELD)
# =========================

plt.figure(figsize=(10, 8))

# Campo vectorial (predicción del modelo)
plt.quiver(X1_mesh, X2_mesh, Y1_pred, Y2_pred, color='blue', alpha=0.6)

# Datos reales (opcional: también como vectores)
plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], color='red', alpha=0.8)

plt.title('Vector Field: F(x1, x2) = (y1, y2)')
plt.xlabel('X1')
plt.ylabel('X2')

plt.grid()
plt.show()
