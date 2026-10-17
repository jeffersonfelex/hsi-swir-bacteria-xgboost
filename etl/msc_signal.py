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
    ref_centered = ref - np.mean(ref)
    ref_var = np.sum(ref_centered ** 2)
    if ref_var < 1e-10:
        return data.copy()

    data_centered = data - np.mean(data, axis=1, keepdims=True)
    slope = np.dot(data_centered, ref_centered) / ref_var
    intercept = np.mean(data, axis=1) - slope * np.mean(ref)

    slope_matrix = slope[:, np.newaxis]
    intercept_matrix = intercept[:, np.newaxis]

    safe_slope = np.where(np.abs(slope_matrix) < 1e-10, 1.0, slope_matrix)
    corrected = (data - intercept_matrix) / safe_slope
    return np.where(np.abs(slope_matrix) < 1e-10, data, corrected)


def run(output_dir: str, folder_name: str, input_filename: str = None):
    """
    Lê o CSV do ROI filtrado, aplica MSC e salva o resultado.

    Args:
        output_dir: diretório onde estão os arquivos
        folder_name: nome base da imagem
        input_filename: nome específico do arquivo de entrada (opcional)
    """
    if input_filename:
        input_path = os.path.join(output_dir, input_filename)
    else:
        candidates = [
            os.path.join(output_dir, f"{folder_name}_roi_filtrado.csv"),
            os.path.join(output_dir, f"{folder_name}_roi.csv"),
        ]
        input_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

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