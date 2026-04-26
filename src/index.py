import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# 1. DATA LOADING FROM CSV
# =========================

try:
    df = pd.read_csv('src/data.csv')
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found.")
    exit()

# Extraemos X1, X2 y Y
# Combinamos x1 y x2 en una sola matriz de características (Features)
X = df[['x1', 'x2']].values
Y = df['y'].values.reshape(-1, 1)

# =========================
# 2. DATA NORMALIZATION
# =========================

# Normalizamos ambas columnas de X y la de Y
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(), Y.std()

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# =========================
# 3. MODEL DEFINITION
# =========================

model = tf.keras.Sequential(
    [
        # Cambiamos input_shape a (2,) porque ahora entran x1 y x2
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

# Para visualizar en 3D, creamos una malla (mesh) de puntos
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

# Aplanamos la malla para pasarla por el modelo
X_test = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]
X_test_norm = (X_test - X_mean) / X_std

Y_pred_norm = model.predict(X_test_norm)
Y_pred = Y_pred_norm * Y_std + Y_mean
Y_pred_mesh = Y_pred.reshape(X1_mesh.shape)

# =========================
# 6. VISUALIZATION (3D)
# =========================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficamos los puntos reales
ax.scatter(
    X[:, 0],
    X[:, 1],
    Y.flatten(),
    color='blue',
    label='Actual Data',
    alpha=1,
)

# Graficamos la superficie de predicción
surf = ax.plot_surface(
    X1_mesh, X2_mesh, Y_pred_mesh, color='red', alpha=0.4, label='Model Prediction'
)

ax.set_title('Neural Network Fit: Y = f(X1, X2)')
ax.set_xlabel('X1 Axis')
ax.set_ylabel('X2 Axis')
ax.set_zlabel('Y Axis')

plt.show()
