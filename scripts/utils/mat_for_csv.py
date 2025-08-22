from scipy.io import loadmat
import pandas as pd
import os

output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed"
output_path = os.path.join(output_dir, "dados_convertidos_mat_para_csv.csv")

data = loadmat("/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_raw/casca_marmelo.mat")

casca_maca = data['casca_marmelo']
n_amostras_n_bandas = casca_maca.reshape(-1, casca_maca.shape[2])

df = pd.DataFrame(n_amostras_n_bandas)
df.columns = [f'Banda_{i}' for i in range(df.shape[1])]
df.to_csv(output_path, index=False)