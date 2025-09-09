# indices_espectrais.py

import pandas as pd
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/feature_engineering/espectros_derivadasLocais_bandasSelecionadas.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/feature_engineering"
output_path = os.path.join(output_dir, "data_indices_espectrais.csv")

df = pd.read_csv(input_path)

indices = pd.DataFrame(index=df.index)

if df.shape[1] >= 2:
    indices["indice1"] = (df.iloc[:, 1] - df.iloc[:, 0]) / (df.iloc[:, 1] + df.iloc[:, 0] + 1e-9)
    indices["indice2"] = df.iloc[:, 1] / (df.iloc[:, 0] + 1e-9)

if df.shape[1] >= 3:
    indices["indice3"] = df.iloc[:, 2] - df.iloc[:, 0]

data_features_final = pd.concat([df, indices], axis=1)

data_features_final.to_csv(output_path, index=False)
print("Arquivo salvo com sucesso em:\n", output_path)
