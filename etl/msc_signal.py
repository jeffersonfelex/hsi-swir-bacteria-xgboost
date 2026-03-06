import os
import pandas as pd
import numpy as np

KEEP_INTERMEDIATE = False


def msc_signal(data: np.ndarray) -> np.ndarray:
    """
    Aplica Multiplicative Scatter Correction (MSC).

    Corrige variações de espalhamento e iluminação ajustando cada espectro
    em relação ao espectro de referência (média de todos os espectros).

    Args:
        data: matriz [n_samples × n_bands]

    Returns:
        matriz corrigida [n_samples × n_bands]
    """
    ref = np.mean(data, axis=0)
    corrected = []

    for spectrum in data:
        coeffs = np.polyfit(ref, spectrum, 1)
        slope, intercept = coeffs[0], coeffs[1]

        if abs(slope) < 1e-10:
            corrected.append(spectrum)
        else:
            corrected.append((spectrum - intercept) / slope)

    return np.array(corrected)


def run(output_dir: str, folder_name: str):
    """
    Lê o CSV do ROI filtrado, aplica MSC e salva o resultado.

    Args:
        output_dir: diretório onde estão os arquivos
        folder_name: nome base da imagem
    """
    input_path  = os.path.join(output_dir, f"{folder_name}_roi_filtrado.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_msc_signal.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ ROI não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"❌ ROI vazio: {input_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"❌ Nenhuma coluna numérica em {input_path}")

    data = df[numeric_cols].values

    if data.shape[0] < 2:
        raise ValueError(f"❌ MSC requer pelo menos 2 amostras. Encontrado: {data.shape[0]}")

    corrected = msc_signal(data)

    pd.DataFrame(corrected, columns=numeric_cols).to_csv(output_path, index=False)
    print(f"✅ MSC aplicado: {output_path}")
    print(f"   Dimensões: {corrected.shape[0]} amostras × {corrected.shape[1]} bandas")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Removido: {input_path}")