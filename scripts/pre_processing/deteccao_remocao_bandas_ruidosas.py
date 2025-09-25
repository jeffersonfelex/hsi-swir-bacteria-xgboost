import pandas as pd
import numpy as np
import os

input_path = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing/espectros_derivada_primeira_ordem.csv"
output_dir = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"
output_path = os.path.join(output_dir, 'espectros_deteccao_remocao_bandas_ruidosas.csv')

df = pd.read_csv(input_path)
data = df.values

# --- ANÁLISE DE VARIÂNCIA ---
variancias = np.var(data, axis=0)

# --- CRITÉRIO 1: EXTREMIDADES ESPECTRAIS ---
percentual_extremidades = 0.10  # 10% das bandas em cada extremidade
n_extremidades = int(data.shape[1] * percentual_extremidades)

# --- CRITÉRIO 2: LIMIAR DE VARIÂNCIA ---
limiar_variancia = 0.05 * np.max(variancias)  # 5% do máximo
limiar_centro = 0.01 * np.max(variancias)     # 1% do máximo (mais rigoroso para centro)

# --- IDENTIFICAÇÃO AUTOMÁTICA DE BANDAS PROBLEMÁTICAS ---
bandas_problematicas = []

# Bandas das extremidades com variância muito baixa
for i in range(n_extremidades):
    if variancias[i] < limiar_variancia:
        bandas_problematicas.append(i)
    if variancias[-(i+1)] < limiar_variancia:
        bandas_problematicas.append(data.shape[1] - (i+1))

# Bandas do centro com variância extremamente baixa
for i in range(n_extremidades, data.shape[1] - n_extremidades):
    if variancias[i] < limiar_centro:
        bandas_problematicas.append(i)

bandas_problematicas = sorted(set(bandas_problematicas))  # Remover duplicatas

# --- APLICAÇÃO AUTOMÁTICA DA FILTRAGEM ---
data_filtrada = np.delete(data, bandas_problematicas, axis=1)
colunas_filtradas = [c for i, c in enumerate(df.columns) if i not in bandas_problematicas]


df_filtrado = pd.DataFrame(data_filtrada, columns=colunas_filtradas)
df_filtrado.to_csv(output_path, index=False)

print("Arquivo com bandas filtradas salvo com sucesso em:\n", output_path)