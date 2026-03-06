import os
import pandas as pd
import numpy as np

KEEP_INTERMEDIATE = False


def log_transform(data: np.ndarray) -> np.ndarray:
    """
    Aplica transformada logarítmica Log(1/R) = -Log(R).

    Realça bandas de baixa intensidade e comprime amplitudes,
    facilitando a modelagem de relações não lineares.

    Args:
        data: matriz [n_samples × n_bands]

    Returns:
        matriz transformada [n_samples × n_bands]
    """
    data_clean = data.copy()
    data_clean[data_clean <= 0]        = 1e-6
    data_clean[np.isnan(data_clean)]   = 1e-6
    data_clean[np.isinf(data_clean)]   = 1e-6

    return -np.log(data_clean)


def run(output_dir: str, folder_name: str):
    """
    Lê o CSV do ROI filtrado, aplica Log(1/R) e salva o resultado.

    Args:
        output_dir: diretório onde estão os arquivos
        folder_name: nome base da imagem
    """
    input_path  = os.path.join(output_dir, f"{folder_name}_roi_filtrado.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_log.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ ROI não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"❌ ROI vazio: {input_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"❌ Nenhuma coluna numérica em {input_path}")

    data = df[numeric_cols].values
    transformed = log_transform(data)

    pd.DataFrame(transformed, columns=numeric_cols).to_csv(output_path, index=False)
    print(f"✅ Log(1/R) aplicado: {output_path}")
    print(f"   Dimensões: {transformed.shape[0]} amostras × {transformed.shape[1]} bandas")
    print(f"   Range: [{transformed.min():.3f}, {transformed.max():.3f}]")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Removido: {input_path}")