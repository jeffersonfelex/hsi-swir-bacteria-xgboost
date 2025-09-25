import numpy as np
import pandas as pd
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_msc.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"
output_path = os.path.join(output_dir, 'espectros_log.csv')

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)
data = df.values


data[data <= 0] = 1e-6 # Evita log de zero ou valores negativos
data_log = -np.log(data)

df_espectros = pd.DataFrame(data_log, columns=df.columns)

df_espectros.to_csv(output_path, index=False)
print("Arquivo corrigido salvo com sucesso em:\n", output_path)