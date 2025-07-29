import pandas as pd
from scipy.signal import savgol_filter
import os

def apply_derivative(input_path: str, output_path: str,window_length=15, polyorder=2, deriv=1):

    df = pd.read_csv(input_path)

    if window_length >= df.shape[1]:
        raise ValueError("window_length não pode ser maior ou igual ao número de colunas (bandas).")

    derivada = savgol_filter(df.values, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
    df_deriv = pd.DataFrame(derivada, columns=df.columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_deriv.to_csv(output_path, index=False)
    print(f"Derivada salva em: {output_path}")
