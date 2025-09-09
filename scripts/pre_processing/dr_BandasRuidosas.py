import pandas as pd
import numpy as np
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_derivada_PrimeiraOrdem.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"
output_path = os.path.join(output_dir, 'espectros_bandas_filtradas.csv')

df = pd.read_csv(input_path)
data = df.values

variancias = np.var(data, axis=0)
limiar = 0.1 * np.max(variancias)
bandas_ruidosas = np.where(variancias < limiar)[0]

data_filtrada = np.delete(data, bandas_ruidosas, axis=1)
colunas_filtradas = [c for i, c in enumerate(df.columns) if i not in bandas_ruidosas]

df_filtrado = pd.DataFrame(data_filtrada, columns=colunas_filtradas)
df_filtrado.to_csv(output_path, index=False)

print("Arquivo com bandas filtradas salvo com sucesso em:\n", output_path)