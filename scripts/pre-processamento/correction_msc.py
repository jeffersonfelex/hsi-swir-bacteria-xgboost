import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

input_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_convertidos_mat_para_csv.csv"
output_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_corregidos_msc.csv"

def apply_msc(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    data = df.values

    # Calcula o espectro de referência (média das amostras)
    espectro_referencia = np.mean(data, axis=0).reshape(-1, 1)
    bandas_corrigidas = np.zeros_like(data)

    for i in range(data.shape[0]):
        amostra = data[i, :].reshape(-1, 1)
        modelo = LinearRegression().fit(espectro_referencia, amostra)
        inclinacao = modelo.coef_[0][0]
        intercepto = modelo.intercept_[0]
        bandas_corrigidas[i, :] = (data[i, :] - intercepto) / inclinacao

    df_corrigido = pd.DataFrame(bandas_corrigidas, columns=df.columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_corrigido.to_csv(output_path, index=False)

# Executa
apply_msc(input_csv, output_csv)
print("✅ Arquivo corrigido salvo com sucesso em:\n", output_csv)
