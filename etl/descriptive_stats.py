import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_roi.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")

    # Se ROI não existir ou estiver vazia, cria arquivo vazio
    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Pulando features descritivas.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {folder_name}: ROI vazia. Criando arquivo de features vazio.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df.values
    n_bandas = data.shape[1]

    # --- Estatísticas por banda ---
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    kurts = kurtosis(data, axis=0, nan_policy='omit')
    skews = skew(data, axis=0, nan_policy='omit')

    # --- Montagem do dataframe final ---
    features = {}
    for i in range(n_bandas):
        features[f"mean_b{i}"] = [means[i]]
        features[f"std_b{i}"] = [stds[i]]
        features[f"kurt_b{i}"] = [kurts[i]]
        features[f"skew_b{i}"] = [skews[i]]

    features_df = pd.DataFrame(features)
    features_df.to_csv(output_path, index=False)
    print(f"✅ Features descritivas geradas: {output_path}")
    print(f"ℹ️ Bandas: {n_bandas} | Pixels: {data.shape[0]}")

    # Remove arquivo anterior para não acumular dados brutos (opcional)
    try:
        os.remove(input_path)
        print(f"🧹 Arquivo bruto removido: {input_path}")
    except Exception:
        pass
