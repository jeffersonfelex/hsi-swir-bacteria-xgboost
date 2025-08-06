import pandas as pd
from scipy.signal import savgol_filter
import os


input_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_transformada_log.csv"
output_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_derivada_primeira_ordem.csv"

def aplicar_derivada_primeira_ordem(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    X = df.values

    # Aplica a derivada de primerida ordem com Savitzky-Golay
    derivada = savgol_filter(X, window_length=11, polyorder=2, deriv=1)

    df_derivada = pd.DataFrame(derivada, columns=df.columns)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_derivada.to_csv(output_path, index=False)

    print("✅ Derivada de primeira ordem aplicada e salva em:")
    print(output_path)

aplicar_derivada_primeira_ordem(input_csv, output_csv)
