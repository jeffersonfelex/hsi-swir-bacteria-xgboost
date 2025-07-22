import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os

output_dir = "C:\Users\jefferson.faria\Documents\GitHub\ic-2025\ETL\data_processed\derivada_po"
output_path = os.path.join(output_dir, "data_samples_correction_msc.csv")

# Carregar os dados
data = pd.read_csv(r'C:\Users\jefferson.faria\Documents\GitHub\ic-2025\ETL\data_processed\data_processed_transf_log\data_samples_correction_msc.csv', index_col= 0 )

# Suponha que cada linha seja uma amostra e cada coluna uma variável espectral
# Aplicar derivada de 1ª ordem (eixo=1: por linha)
data_deriv = savgol_filter(data.values, window_length=15, polyorder=2, deriv=1, axis=1)

# Converter de volta em DataFrame, mantendo índice e colunas
data_deriv_df = pd.DataFrame(data_deriv, index=data.index, columns=data.columns)

data_deriv_df.to_csv(output_path)

# Plot de comparação (1 espectro como exemplo)
idx = 0  # índice da amostra
plt.plot(data.columns.astype(float), data.iloc[idx], label='Original')
plt.plot(data.columns.astype(float), data_deriv_df.iloc[idx], label='Derivada 1ª Ordem')
plt.xlabel('Comprimento de onda')
plt.ylabel('Absorbância')
plt.legend()
plt.title('Savitzky-Golay - Derivada 1ª Ordem')
plt.grid(True)
plt.show()
