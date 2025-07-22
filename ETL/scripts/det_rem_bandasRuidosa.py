import pandas as pd
import matplotlib.pyplot as plt
import os


output_dir = "C:\Users\jefferson.faria\Documents\GitHub\ic-2025\ETL\data_processed\det_rem_bandasRuidosa"
output_path = os.path.join(output_dir, "data_samples_correction_msc.csv")

# Carregar os dados
data = pd.read_csv(r'C:\Users\jefferson.faria\Documents\GitHub\ic-2025\ETL\data_processed\data_processed_transf_log\data_samples_correction_msc.csv', index_col=0)

# Calcular variância por coluna (banda)
variancia = data.var(axis=0)

# Configurar limiar de variância
limiar = 1e-4

# Aplicar filtro combinado: variância alta + região espectral confiável
colunas_float = data.columns.astype(float)
filtro_combinado = (variancia > limiar) & (colunas_float > 750) & (colunas_float < 2400)
data_filtrado = data.loc[:, filtro_combinado]

df = pd.DataFrame(data_filtrado)

df.to_csv(output_path)
# Plotar variância original
plt.figure(figsize=(10, 4))
plt.plot(colunas_float, variancia.values, label='Variância por banda')
plt.axhline(y=limiar, color='red', linestyle='--', label='Limiar de variância')
plt.xlabel('Comprimento de onda')
plt.ylabel('Variância')
plt.title('Análise de Variância Espectral')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
