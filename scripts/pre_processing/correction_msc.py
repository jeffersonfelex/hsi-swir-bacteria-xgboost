import pandas as pd
import os
from chemotools.scatter import MultiplicativeScatterCorrection

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_normalizados.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'espectros_msc.csv')

df = pd.read_csv(input_path)
data = df.values


msc = MultiplicativeScatterCorrection()
bandas_corrigidas = msc.fit_transform(data)

df_espectros = pd.DataFrame(bandas_corrigidas, columns=df.columns)
df_espectros.to_csv(output_path, index=False)

print(f"\n✅ Arquivo MSC salvo com sucesso em: {output_path}")
print(f"✅ Total de amostras processadas: {df_espectros.shape[0]}")
