import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.linear_model import LinearRegression

output_dir = "./ETL/data_processed/data_processed_correction_msc"
output_path = os.path.join(output_dir, "data_samples_correction_msc.csv")


df = pd.read_csv(r'C:\Users\jefferson.faria\Documents\GitHub\ic-2025\ETL\data_processed\data_processed_csv\data_samples_processed.csv')
data = df.values


espectro_referencia = np.mean(data, axis = 0).reshape(-1,1)
dados_corrigidos = np.zeros_like(data)
for i in range(data.shape[0]):
    sample = data[i, :].reshape(-1, 1)
    
    model = LinearRegression().fit(espectro_referencia, sample)
    a = model.coef_[0][0]
    b = model.intercept_[0]
    
    corrected_sample = (data[i, :] - b) / a
    dados_corrigidos[i, :] = corrected_sample

df = pd.DataFrame(dados_corrigidos)
df.to_csv(output_path)