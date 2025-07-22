import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "./ETL/data_processed/data_processed_transf_log"
output_path = os.path.join(output_dir, "data_samples_correction_msc.csv")

data = pd.read_csv(r'C:\Users\jefferson.faria\Documents\GitHub\ic-2025\ETL\data_processed\data_processed_correction_msc\data_samples_correction_msc.csv')

# Separar apenas os dados espectrais (supondo que não tenha colunas categóricas)
X = data.values

# Verifica se há zeros (log(0) é indefinido) e corrige
X[X <= 0] = 1e-6  # evita log de zero ou negativo

# Aplicar Transformada Logarítmica (log(1/R) = -log(R))
X_log = -np.log(X)
df = pd.DataFrame(X_log)

df.to_csv(output_path)


idx = 0  # índice da amostra a visualizar
plt.figure(figsize=(10, 5))
plt.plot(X[idx], label="Original (Reflectância)")
plt.plot(X_log[idx], label="Log(1/R) (Absorbância)", linestyle="--")
plt.legend()
plt.title(f"Transformada Log(1/R) - Amostra {idx}")
plt.xlabel("Comprimento de Onda (ou índice)")
plt.ylabel("Intensidade")
plt.grid(True)
plt.show()
