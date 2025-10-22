import os
import numpy as np
import pandas as pd

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_filtrado.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_index.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df.values
    n_bandas = data.shape[1]

    if n_bandas < 3:
        # Não há bandas suficientes para índices derivados
        print(f"⚠️ Bandas insuficientes ({n_bandas}) em {folder_name}. Gerando saída vazia.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Média por banda (1 espectro médio)
    mean_spectrum = np.mean(data, axis=0)

    # Seleciona bandas com base em heurística:
    idx_max = np.argmax(mean_spectrum)
    idx_min = np.argmin(mean_spectrum)
    idx_mid = n_bandas // 2  # banda central

    b_max = mean_spectrum[idx_max]
    b_min = mean_spectrum[idx_min]
    b_mid = mean_spectrum[idx_mid]

    # Índices empíricos
    idx_ratio_max_min = (b_max - b_min) / (b_max + b_min + 1e-8)
    idx_ratio_max_mid = (b_max - b_mid) / (b_max + b_mid + 1e-8)
    idx_ratio_mid_min = (b_mid - b_min) / (b_mid + b_min + 1e-8)

    idx_diff_max_min = b_max - b_min
    idx_diff_max_mid = b_max - b_mid
    idx_diff_mid_min = b_mid - b_min

    idx_prod_max_min = b_max * b_min
    idx_prod_max_mid = b_max * b_mid
    idx_prod_mid_min = b_mid * b_min

    features = {
        "ratio_max_min": idx_ratio_max_min,
        "ratio_max_mid": idx_ratio_max_mid,
        "ratio_mid_min": idx_ratio_mid_min,
        "diff_max_min": idx_diff_max_min,
        "diff_max_mid": idx_diff_max_mid,
        "diff_mid_min": idx_diff_mid_min,
        "prod_max_min": idx_prod_max_min,
        "prod_max_mid": idx_prod_max_mid,
        "prod_mid_min": idx_prod_mid_min,
        "band_max_idx": idx_max,
        "band_min_idx": idx_min,
        "band_mid_idx": idx_mid
    }

    pd.DataFrame([features]).to_csv(output_path, index=False)
    print(f"✅ Índices espectrais empíricos salvos: {output_path}")
