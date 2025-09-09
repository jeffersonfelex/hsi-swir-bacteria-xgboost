import pandas as pd
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/feature_engineering/espectros_estatisticas_Descritivas.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/feature_engineering"
output_path = os.path.join(output_dir, 'espectros_derivadasLocais_bandasSelecionadas.csv')

df = pd.read_csv(input_path)

derivadas = df.diff(axis=1).iloc[:, 1:]
derivadas.columns = [f"deriv_{col}" for col in derivadas.columns]

indices_para_razao = [(0, 1), (1, 2)]
razoes = pd.DataFrame(index=df.index)

for i, j in indices_para_razao:
    razoes[f"razao_col{i}_col{j}"] = df.iloc[:, i] / df.iloc[:, j]

data_features = pd.concat([derivadas, razoes], axis=1)

data_features.to_csv(output_path, index=False)
print("Arquivo corrigido salvo com sucesso em:\n", output_path)


