from scipy.io import loadmat
import pandas as pd
import os

data = loadmat("./ETL/data_raw/casca_maca.mat")

casca_maca = data['casca_maca']
n_amostras_n_bandas = casca_maca.reshape(-1, casca_maca.shape[2])

df = pd.DataFrame(n_amostras_n_bandas)

output_dir = "./ETL/data_processed"
output_path = os.path.join(output_dir, "data_samples_processed.csv")

df.to_csv(output_path, index=False)
