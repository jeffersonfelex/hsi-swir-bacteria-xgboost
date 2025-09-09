from scipy.signal import savgol_filter
import pandas as pd
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_transformada_log.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"
output_path = os.path.join(output_dir, 'espectros_derivada_PrimeiraOrdem.csv')

df = pd.read_csv(input_path)
data = df.values

# Aplica a derivada de primerida ordem com Savitzky-Golay
data_derivada = savgol_filter(data, window_length=11, polyorder=2, deriv=1)

df_espectros = pd.DataFrame(data_derivada, columns=df.columns)


df_espectros.to_csv(output_path, index=False)
print("Arquivo corrigido salvo com sucesso em:\n", output_path)