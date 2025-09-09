from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np
import os


input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_bandas_filtradas.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/feature_engineering"
output_path = os.path.join(output_dir, 'espectros_estatisticas_Descritivas.csv')

df = pd.read_csv(input_path)
data = df.values


stats = pd.DataFrame(index=df.index)
stats['media'] = np.mean(data, axis=1)
stats['desvio_padrao'] = np.std(data, axis=1, ddof=1)  # ddof=1 para amostra
stats['curtose'] = [kurtosis(row, fisher=True) for row in data]  # Fisher=True -> normal=0
stats['skewness'] = [skew(row) for row in data]


stats.to_csv(output_path, index=False)
print("Arquivo de estatísticas descritivas salvo com sucesso em:\n", output_path)
