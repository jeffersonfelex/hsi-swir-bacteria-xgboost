import os
import numpy as np
import pandas as pd

KEEP_INTERMEDIATE = False


def spectral_features(data: np.ndarray) -> pd.DataFrame:
    """
    Extrai índices espectrais empíricos de cada espectro.

    Índices baseados em regiões SWIR características de componentes bacterianos
    (lipídios, proteínas, polissacarídeos).

    Args:
        data: matriz [n_samples × n_bands]

    Returns:
        DataFrame [n_samples × n_features] com prefixo 'spec_'
    """
    n_samples, n_bands = data.shape

    if n_bands < 10:
        raise ValueError(f"❌ Bandas insuficientes ({n_bands} < 10) para spectral_features.")

    # Índices de banda por região espectral (proporcionais ao total de bandas)
    idx_1200 = min(int(n_bands * 0.13), n_bands - 1)
    idx_1450 = min(int(n_bands * 0.30), n_bands - 1)
    idx_1700 = min(int(n_bands * 0.47), n_bands - 1)
    idx_2100 = min(int(n_bands * 0.73), n_bands - 1)
    idx_2300 = min(int(n_bands * 0.87), n_bands - 1)

    b_1200 = data[:, idx_1200]
    b_1450 = data[:, idx_1450]
    b_1700 = data[:, idx_1700]
    b_2100 = data[:, idx_2100]
    b_2300 = data[:, idx_2300]

    max_vals = np.max(data, axis=1)

    # Segunda derivada para curvatura
    second_deriv = np.gradient(np.gradient(data, axis=1), axis=1)

    features = {
        # Normalized Difference Indices
        "spec_ndi_lipid_protein":   (b_1200 - b_2100) / (b_1200 + b_2100 + 1e-8),
        "spec_ndi_water_lipid":     (b_1450 - b_1700) / (b_1450 + b_1700 + 1e-8),
        "spec_ndi_protein_polysac": (b_2100 - b_2300) / (b_2100 + b_2300 + 1e-8),
        # Simple Ratios
        "spec_sr_lipid_protein":    b_1200 / (b_2100 + 1e-8),
        "spec_sr_water_lipid":      b_1450 / (b_1700 + 1e-8),
        "spec_sr_protein_polysac":  b_2100 / (b_2300 + 1e-8),
        # Profundidades relativas
        "spec_depth_1200":          b_1200 / (max_vals + 1e-8),
        "spec_depth_1450":          b_1450 / (max_vals + 1e-8),
        "spec_depth_1700":          b_1700 / (max_vals + 1e-8),
        "spec_depth_2100":          b_2100 / (max_vals + 1e-8),
        "spec_depth_2300":          b_2300 / (max_vals + 1e-8),
        # Índices compostos
        "spec_composite_lipid":     (b_1200 + b_1700) / (2 * max_vals + 1e-8),
        "spec_composite_protein":   (b_1450 + b_2100) / (2 * max_vals + 1e-8),
        "spec_composite_polysac":   b_2300 / (max_vals + 1e-8),
        # Posição do pico
        "spec_peak_position":       np.argmax(data, axis=1) / n_bands,
        # Curvatura
        "spec_curvature_1450":      second_deriv[:, idx_1450],
        "spec_curvature_2100":      second_deriv[:, idx_2100],
    }

    return pd.DataFrame(features)


def run(output_dir: str, folder_name: str, suffix: str = "derivada"):
    """
    Lê um CSV de espectros, aplica spectral_features e salva o resultado.
    """
    input_path  = os.path.join(output_dir, f"{folder_name}_{suffix}.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"❌ Arquivo vazio: {input_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"❌ Nenhuma coluna numérica em {input_path}")

    feats = spectral_features(df[numeric_cols].values)
    feats.to_csv(output_path, index=False)

    print(f"✅ spectral_features extraídas: {output_path}")
    print(f"   Dimensões: {feats.shape[0]} amostras × {feats.shape[1]} features")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Removido: {input_path}")