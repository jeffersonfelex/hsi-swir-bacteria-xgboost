import numpy as np
import pandas as pd
import os

input_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_corregidos_msc.csv"
output_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_transformada_log.csv"

def apply_log_transform(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    data = df.values

    # Evita log de zero ou valores negativos
    data[data <= 0] = 1e-6
    data_log = -np.log(data)

    df_log = pd.DataFrame(data_log, columns=df.columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_log.to_csv(output_path, index=False)

# Executa
apply_log_transform(input_csv, output_csv)
print("✅ Transformação log aplicada e salva com sucesso em:\n", output_csv)
