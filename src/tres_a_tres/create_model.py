import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tres_a_tres.create_data import create_data

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
X = df[['x1', 'x2', 'x3']].values
Y = df[['y1', 'y2', 'y3']].values

# =========================
# 2. DATA NORMALIZATION
# =========================

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)

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
        tf.keras.Input(shape=(3,)),  # ✅ corregido
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Dense(3),
    ]
)

model.compile(optimizer='adam', loss='mse')

# =========================
# 4. TRAINING
# =========================

print('Training the model with x1, x2, x3 predicting y1, y2, y3...')
model.fit(X_norm, Y_norm, epochs=200, verbose=0)

# =========================
# 5. PREDICTION (GRID)
# =========================

# 🔥 menos resolución para que sea visible
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 5)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 5)
x3_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 5)

X1_mesh, X2_mesh, X3_mesh = np.meshgrid(x1_range, x2_range, x3_range)

X_test = np.c_[X1_mesh.ravel(), X2_mesh.ravel(), X3_mesh.ravel()]
X_test_norm = (X_test - X_mean) / X_std

Y_pred_norm = model.predict(X_test_norm, verbose=0)
Y_pred = Y_pred_norm * Y_std + Y_mean

# Separar outputs
Y1_pred = Y_pred[:, 0].reshape(X1_mesh.shape)
Y2_pred = Y_pred[:, 1].reshape(X1_mesh.shape)
Y3_pred = Y_pred[:, 2].reshape(X1_mesh.shape)

# Guardar modelo
model.save('model.keras')

# =========================
# 6. VISUALIZATION (SEPARATED)
# =========================

fig = plt.figure(figsize=(16, 7))

# ----------- Gráfico 1: Predicción del modelo -----------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

U1 = Y1_pred - X1_mesh
U2 = Y2_pred - X2_mesh
U3 = Y3_pred - X3_mesh

ax1.quiver(X1_mesh, X2_mesh, X3_mesh, U1, U2, U3, color='blue', alpha=0.6)

ax1.set_title('Predicted Vector Field')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('X3')

# ----------- Gráfico 2: Datos reales -----------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

U = Y - X
ax2.quiver(X[:, 0], X[:, 1], X[:, 2], U[:, 0], U[:, 1], U[:, 2], color='red', alpha=0.6)

ax2.set_title('Real Vector Field')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('X3')

plt.tight_layout()
plt.show()
