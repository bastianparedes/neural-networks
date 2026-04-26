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
    # 2. Data Generation
    # ==========================================
    n_samples = 100

    # Variables independientes
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    x3 = np.random.uniform(-5, 5, n_samples)

    # Nuevas variables dependientes (intercambio)
    y1 = -x2
    y2 = x1
    y3 = x3

    # ==========================================
    # 3. DataFrame Creation and Export
    # ==========================================
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y1': y1, 'y2': y2, 'y3': y3})

    df.to_csv(file_name, index=False)

    print(f"Archivo '{file_name}' generado con éxito.")
    print(df.head())
