import os
import numpy as np
import pandas as pd

KEEP_INTERMEDIATE = False  # ❌ Remove derivada após extrair features

def run(output_dir: str, folder_name: str):
    """
    Extrai features espectrais de cada espectro.
    
    Features calculadas:
    - Gradiente médio (derivada local média)
    - Gradiente máximo (máxima variação entre bandas)
    - Razão primeira/última banda
    - Razão bandas quartis (Q1/Q3)
    - Área sob a curva espectral (integral)
    - Amplitude espectral (max - min)
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo
    """
    input_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

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
    n_samples, n_bands = data.shape

    # Verificar se há bandas suficientes
    if n_bands < 2:
        print(f"⚠️ Bandas insuficientes ({n_bands} < 2) para features espectrais")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # ========================================
    # CALCULAR FEATURES ESPECTRAIS
    # ========================================

    features_dict = {}

    # 1. Gradiente médio (derivada local média)
    gradients = np.gradient(data, axis=1)  # Derivada ao longo das bandas
    features_dict['grad_mean'] = np.mean(np.abs(gradients), axis=1)

    # 2. Gradiente máximo (máxima variação entre bandas)
    features_dict['grad_max'] = np.max(np.abs(gradients), axis=1)

    # 3. Razão primeira/última banda
    b_first = data[:, 0]
    b_last = data[:, -1]
    features_dict['ratio_first_last'] = (b_last - b_first) / (b_last + b_first + 1e-8)

    # 4. Razão entre quartis (Q1/Q3)
    if n_bands >= 4:
        q1_idx = n_bands // 4
        q3_idx = 3 * n_bands // 4
        b_q1 = data[:, q1_idx]
        b_q3 = data[:, q3_idx]
        features_dict['ratio_q1_q3'] = (b_q3 - b_q1) / (b_q3 + b_q1 + 1e-8)
    else:
        features_dict['ratio_q1_q3'] = np.zeros(n_samples)

    # 5. Área sob a curva (integral - soma trapezoidal)
    features_dict['area_under_curve'] = np.trapz(data, axis=1)

    # 6. Amplitude espectral (range)
    features_dict['spectral_range'] = np.max(data, axis=1) - np.min(data, axis=1)

    # 7. Posição do pico máximo (índice da banda com maior valor)
    features_dict['peak_position'] = np.argmax(data, axis=1) / n_bands  # Normalizado [0,1]

    # 8. Largura espectral efetiva (número de bandas acima de 50% do máximo)
    max_vals = np.max(data, axis=1, keepdims=True)
    threshold = 0.5 * max_vals
    above_threshold = (data >= threshold).sum(axis=1)
    features_dict['spectral_width'] = above_threshold / n_bands  # Normalizado

    # Criar DataFrame com features
    features_df = pd.DataFrame(features_dict)

    # Salvar resultado
    features_df.to_csv(output_path, index=False)
    
    print(f"✅ Features espectrais extraídas: {output_path}")
    print(f"   Dimensões: {features_df.shape[0]} amostras × {features_df.shape[1]} features")
    print(f"   Features: {list(features_df.columns)}")

    # Limpar arquivo intermediário
    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")