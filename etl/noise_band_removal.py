import numpy as np
import pandas as pd
import os

KEEP_INTERMEDIATE = False  # ❌ Remove Derivada após filtragem

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_filtrado.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df.values
    variancias = np.var(data, axis=0)
    n_ext = int(data.shape[1] * 0.10)
    lim_ext = 0.05 * np.max(variancias)
    lim_centro = 0.01 * np.max(variancias)

    bandas_problematicas = []
    for i in range(n_ext):
        if variancias[i] < lim_ext:
            bandas_problematicas.append(i)
        if variancias[-(i+1)] < lim_ext:
            bandas_problematicas.append(data.shape[1] - (i+1))

    for i in range(n_ext, data.shape[1] - n_ext):
        if variancias[i] < lim_centro:
            bandas_problematicas.append(i)

    bandas_problematicas = sorted(set(bandas_problematicas))
    data_filtrada = np.delete(data, bandas_problematicas, axis=1)
    colunas_filtradas = [c for i, c in enumerate(df.columns) if i not in bandas_problematicas]

    pd.DataFrame(data_filtrada, columns=colunas_filtradas).to_csv(output_path, index=False)
    print(f"✅ Bandas ruidosas removidas: {output_path}")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")
