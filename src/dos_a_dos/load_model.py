import os
import pickle

import numpy as np
import tensorflow as tf

# =========================
# 1. UBICACIÓN DEL SCRIPT
# =========================
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# =========================
# 2. CARGAR MODELO
# =========================
model = tf.keras.models.load_model('model.keras')

# =========================
# 3. CARGAR NORMALIZACIÓN
# =========================
with open('scaler.pkl', 'rb') as f:
    X_mean, X_std, Y_mean, Y_std = pickle.load(f)

# =========================
# 4. DATOS DE ENTRADA
# =========================
x1 = 5
x2 = 10

X_new = np.array([[x1, x2]])

# =========================
# 5. NORMALIZAR
# =========================
X_new_norm = (X_new - X_mean) / X_std

# =========================
# 6. PREDICCIÓN
# =========================
Y_pred_norm = model.predict(X_new_norm, verbose=0)

# =========================
# 7. DESNORMALIZAR
# =========================
Y_pred = Y_pred_norm * Y_std + Y_mean

# =========================
# 8. RESULTADO
# =========================
y1 = Y_pred[0][0]
y2 = Y_pred[0][1]

print(f'Predicción para x1={x1}, x2={x2} → y1={y1}, y2={y2}')
