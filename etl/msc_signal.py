import os
import pandas as pd
import numpy as np

KEEP_INTERMEDIATE = False  # ❌ Remove ROI após gerar MSC

def apply_msc(data):
    """
    Aplica Multiplicative Scatter Correction (MSC)
    
    Args:
        data: matriz numpy [n_samples × n_bands]
    
    Returns:
        matriz corrigida numpy [n_samples × n_bands]
    """
    ref = np.mean(data, axis=0)  # Espectro de referência (média)
    corrected = []
    
    for spectrum in data:
        # Ajuste linear: spectrum = slope * ref + intercept
        coeffs = np.polyfit(ref, spectrum, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        # Evitar divisão por zero
        if abs(slope) < 1e-10:
            corrected.append(spectrum)  # Mantém original se slope ~ 0
        else:
            corrected_spectrum = (spectrum - intercept) / slope
            corrected.append(corrected_spectrum)
    
    return np.array(corrected)


def run(output_dir: str, folder_name: str):
    """
    Aplica Multiplicative Scatter Correction (MSC) aos dados do ROI.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo
    """
    input_path = os.path.join(output_dir, f"{folder_name}_roi.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_msc_signal.csv")

    # Verificar se arquivo existe
    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Carregar dados
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Extrair apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        print(f"⚠️ Nenhuma coluna numérica em {input_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df[numeric_cols].values

    # Verificar dimensões mínimas
    if data.shape[0] < 2:
        print(f"⚠️ MSC requer pelo menos 2 amostras. Encontrado: {data.shape[0]}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Aplicar MSC
    corrected = apply_msc(data)

    # Salvar resultado
    df_corrected = pd.DataFrame(corrected, columns=numeric_cols)
    df_corrected.to_csv(output_path, index=False)
    print(f"✅ MSC aplicado: {output_path}")
    print(f"   Dimensões: {corrected.shape[0]} amostras × {corrected.shape[1]} bandas")

    # Limpar arquivo intermediário
    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")