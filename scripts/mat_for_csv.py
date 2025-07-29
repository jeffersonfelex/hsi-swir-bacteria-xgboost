from scipy.io import loadmat
import pandas as pd
import os

data = loadmat("/home/jeffersonfaria/Documentos/ic-2025/ETL/data_raw/casca_maca.mat")


output_dir = "./ETL/data_processed/data_processed_csv"
output_path = os.path.join(output_dir, "data_samples_processed.csv")

casca_maca = data['casca_maca']
n_amostras_n_bandas = casca_maca.reshape(-1, casca_maca.shape[2])

df = pd.DataFrame(n_amostras_n_bandas)
df.to_csv(output_path, index=False)