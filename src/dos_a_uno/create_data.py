import os

import numpy as np
import pandas as pd


def create_data():
    # ==========================================
    # 1. Change Execution Context
    # ==========================================
    # Esto asegura que el archivo se guarde en la misma carpeta que el script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file_name = './data.csv'

    # ==========================================
    # 2. Data Generation (Mathematical Logic)
    # ==========================================
    n_samples = 100

    # Usamos uniform para que x1 y x2 sean independientes y aleatorios
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)

    # Generamos ruido aleatorio
    noise = np.random.normal(0, 1, size=n_samples)

    # Calculamos y sumando x1, x2 y el ruido para que no sea una relación perfecta
    y1 = x1 + x2 + noise

    # ==========================================
    # 3. DataFrame Creation and Export
    # ==========================================
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y1': y1})

    df.to_csv(file_name, index=False)

    print(f"Archivo '{file_name}' generado con éxito.")
    print(
        df.head()
    )  # Muestra las primeras filas para verificar que x1 y x2 son distintos
