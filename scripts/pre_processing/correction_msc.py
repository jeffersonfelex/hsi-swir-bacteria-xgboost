import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/espectros_swir_351_1.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"
output_path = os.path.join(output_dir, 'espectros_correction_msc.csv')

df = pd.read_csv(input_path)
data = df.values

espectro_referencia = np.mean(data, axis=0).reshape(-1, 1)
bandas_corrigidas = np.zeros_like(data)

for i in range(data.shape[0]):
    amostra = data[i, :].reshape(-1, 1)
    modelo = LinearRegression().fit(espectro_referencia, amostra)
    inclinacao = modelo.coef_[0][0]
    intercepto = modelo.intercept_[0]
    bandas_corrigidas[i, :] = (data[i, :] - intercepto) / inclinacao
    
df_espectros = pd.DataFrame(bandas_corrigidas, columns=df.columns)

df_espectros.to_csv(output_path, index=False)
print("Arquivo corrigido salvo com sucesso em:\n", output_path)