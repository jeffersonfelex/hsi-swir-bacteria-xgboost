import numpy as np
import spectral.io.envi as envi
import pandas as pd
import os

input_path = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/hsi_original/351_1_240506-160920/capture/351_1_240506-160920.hdr'
output_dir = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing'
output_path = os.path.join(output_dir, 'espectros_swir_351_1.csv')

img_spectral = envi.open(input_path)

# Se for necessario a leitura das informações da imagem
#print(f"Imagem carregada: {img_spectral}")
#print(f"Dimensões: {img_spectral.shape}")

img_array = img_spectral.load()
print(f"Formato do array: {img_array.shape}")

img_array = np.array(img_array)

altura, largura, n_bandas = img_array.shape
n_amostras = altura * largura

# Reshape para matriz [n_amostras × n_bandas]
espectros_matriz = img_array.reshape(n_amostras, n_bandas)

print(f"Matriz final: {espectros_matriz.shape}")
print(f"n_amostras (pixels): {n_amostras}")
print(f"n_bandas espectrais: {n_bandas}")
print(f"Tipo de dados: {espectros_matriz.dtype}")
print(f"Faixa de valores: [{espectros_matriz.min():.2f}, {espectros_matriz.max():.2f}]")

# Salvar a matriz
# np.save('espectros_swir_351_1.npy', espectros_matriz)

colunas = [f'Banda_{i}' for i in range(n_bandas)]
df_espectros = pd.DataFrame(espectros_matriz, columns=colunas)
df_espectros.to_csv(output_path, index=False)