import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

def apply_msc(input_path: str, output_path: str):
    # Lê o arquivo CSV do caminho passado como parâmetro
    df = pd.read_csv(input_path)
    data = df.values

    # Calcula o espectro referência como a média das amostras (linha média)
    espectro_referencia = np.mean(data, axis=0).reshape(-1, 1)
    dados_corrigidos = np.zeros_like(data)

    for i in range(data.shape[0]):
        sample = data[i, :].reshape(-1, 1)
        model = LinearRegression().fit(espectro_referencia, sample)
        a = model.coef_[0][0]
        b = model.intercept_[0]
        dados_corrigidos[i, :] = (data[i, :] - b) / a

    df_corrigido = pd.DataFrame(dados_corrigidos, columns=df.columns)

    # Garante que a pasta do arquivo de saída exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_corrigido.to_csv(output_path, index=False)
    print(f"MSC aplicado e salvo em: {output_path}")
