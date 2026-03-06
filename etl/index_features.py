import os
import numpy as np
import pandas as pd

KEEP_INTERMEDIATE = False


def index_features(data: np.ndarray) -> pd.DataFrame:
    """
    Extrai índices espectrais por regiões SWIR (lipídios, proteínas, polissacarídeos).

    Args:
        data: matriz [n_samples × n_bands]

    Returns:
        DataFrame [n_samples × n_features] com prefixo 'idx_'
    """
    n_samples, n_bands = data.shape

    if n_bands < 4:
        raise ValueError(f"❌ Bandas insuficientes ({n_bands} < 4) para index_features.")

    # Regiões espectrais por proporção do espectro total
    lipid_region   = slice(0,                n_bands // 3)
    protein_region = slice(n_bands // 3,     2 * n_bands // 3)
    carb_region    = slice(2 * n_bands // 3, n_bands)

    lipid_mean   = np.mean(data[:, lipid_region],   axis=1)
    protein_mean = np.mean(data[:, protein_region], axis=1)
    carb_mean    = np.mean(data[:, carb_region],    axis=1)

    lipid_max = np.max(data[:, lipid_region],   axis=1)
    lipid_min = np.min(data[:, lipid_region],   axis=1)
    prot_max  = np.max(data[:, protein_region], axis=1)
    prot_min  = np.min(data[:, protein_region], axis=1)

    len_lip  = data[:, lipid_region].shape[1]
    len_prot = data[:, protein_region].shape[1]
    len_carb = data[:, carb_region].shape[1]

    features = {
        "idx_lipid_protein_ratio":      lipid_mean   / (protein_mean + 1e-8),
        "idx_protein_carb_ratio":       protein_mean / (carb_mean    + 1e-8),
        "idx_ndi_lip_prot":             (lipid_mean - protein_mean) / (lipid_mean + protein_mean + 1e-8),
        "idx_ndi_prot_carb":            (protein_mean - carb_mean)  / (protein_mean + carb_mean  + 1e-8),
        "idx_lipid_absorption_depth":   lipid_max - lipid_min,
        "idx_protein_absorption_depth": prot_max  - prot_min,
        "idx_lipid_area_norm":          np.trapezoid(data[:, lipid_region],   axis=1) / len_lip,
        "idx_protein_area_norm":        np.trapezoid(data[:, protein_region], axis=1) / len_prot,
        "idx_carb_area_norm":           np.trapezoid(data[:, carb_region],    axis=1) / len_carb,
    }

    return pd.DataFrame(features)


def run(output_dir: str, folder_name: str, suffix: str = "derivada"):
    """
    Lê um CSV de espectros, aplica index_features e salva o resultado.
    """
    input_path  = os.path.join(output_dir, f"{folder_name}_{suffix}.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_index.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"❌ Arquivo vazio: {input_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"❌ Nenhuma coluna numérica em {input_path}")

    feats = index_features(df[numeric_cols].values)
    feats.to_csv(output_path, index=False)

    print(f"✅ index_features extraídas: {output_path}")
    print(f"   Dimensões: {feats.shape[0]} amostras × {feats.shape[1]} features")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Removido: {input_path}")