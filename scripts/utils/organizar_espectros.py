import numpy as np
import spectral.io.envi as envi
import pandas as pd
import os

input_path = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/hsi_original/351_1_240506-160920/capture/351_1_240506-160920.hdr'
output_dir = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing'

img_spectral = envi.open(input_path)
img_array = np.array(img_spectral.load())

# Normalizar se necessário
if img_array.max() > 1.0:
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

altura, largura, n_bandas = img_array.shape
espectros_matriz = img_array.reshape(-1, n_bandas)

colunas = [f'Banda_{i}' for i in range(n_bandas)]
df_espectros = pd.DataFrame(espectros_matriz, columns=colunas)

output_path = os.path.join(output_dir, 'espectros_normalizados.csv')
df_espectros.to_csv(output_path, index=False)
print(f"Dados salvos em: {output_path}")
