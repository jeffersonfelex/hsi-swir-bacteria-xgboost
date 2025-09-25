import pandas as pd
import os
from chemotools.derivative import SavitzkyGolay

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_log.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"

output_path = os.path.join(output_dir, 'espectros_derivada_primeira_ordem.csv')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)
data = df.values

deriv_transform = SavitzkyGolay(window_size=15, polynomial_order=2)
data_derivada = deriv_transform.fit_transform(data)

df_derivada = pd.DataFrame(data_derivada, columns=df.columns)
df_derivada.to_csv(output_path, index=False)
print(f"\nArquivo derivada salvo com sucesso em: {output_path}")