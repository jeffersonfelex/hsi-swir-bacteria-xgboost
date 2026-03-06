import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

KEEP_INTERMEDIATE = False

WINDOW_LENGTH = 15
POLYORDER     = 2
DERIV         = 1


def derivative_transform(data: np.ndarray) -> np.ndarray:
    """
    Aplica derivada de primeira ordem via filtro Savitzky-Golay.

    Parâmetros conforme plano de trabalho:
        - window_length : 15 pontos
        - polyorder     : grau 2
        - deriv         : 1ª ordem

    Destaca inflexões espectrais e minimiza o efeito de baseline.

    Args:
        data: matriz [n_samples × n_bands]

    Returns:
        matriz transformada [n_samples × n_bands]
    """
    n_bands = data.shape[1]
    window  = WINDOW_LENGTH

    if n_bands < window:
        window = n_bands if n_bands % 2 == 1 else n_bands - 1
        print(f"⚠️ window_length ajustado para {window} (n_bands={n_bands})")

    if window < 5:
        raise ValueError(f"❌ Bandas insuficientes para Savitzky-Golay ({n_bands} < 5)")

    # axis=1 → derivada ao longo das bandas (eixo espectral), não das amostras
    return savgol_filter(data, window_length=window, polyorder=POLYORDER, deriv=DERIV, axis=1)


def run(output_dir: str, folder_name: str):
    """
    Lê o CSV do ROI filtrado, aplica Savitzky-Golay e salva o resultado.

    Args:
        output_dir: diretório onde estão os arquivos
        folder_name: nome base da imagem
    """
    input_path  = os.path.join(output_dir, f"{folder_name}_roi_filtrado.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ ROI não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"❌ ROI vazio: {input_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"❌ Nenhuma coluna numérica em {input_path}")

    data        = df[numeric_cols].values
    transformed = derivative_transform(data)

    pd.DataFrame(transformed, columns=numeric_cols).to_csv(output_path, index=False)
    print(f"✅ Savitzky-Golay aplicado: {output_path}")
    print(f"   Dimensões : {transformed.shape[0]} amostras × {transformed.shape[1]} bandas")
    print(f"   Parâmetros: window={WINDOW_LENGTH}, polyorder={POLYORDER}, deriv={DERIV}")
    print(f"   Range     : [{transformed.min():.6f}, {transformed.max():.6f}]")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Removido: {input_path}")