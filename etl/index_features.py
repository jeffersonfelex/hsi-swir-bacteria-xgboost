import os
import numpy as np
import pandas as pd

KEEP_INTERMEDIATE = False  # ❌ Remove derivada após extrair features

def run(output_dir: str, folder_name: str):
    """
    Extrai índices espectrais empíricos de cada espectro.
    
    Índices baseados em regiões espectrais características do SWIR
    para análise de componentes bacterianos (lipídios, proteínas, polissacarídeos).
    
    Features calculadas:
    - NDI (Normalized Difference Index) entre pares de bandas
    - Simple Ratio entre pares de bandas
    - Profundidade de absorção em regiões-chave
    - Razões lipídio/proteína, proteína/polissacarídeo
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo
    """
    input_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_index.csv")

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

    if n_bands < 10:
        print(f"⚠️ Bandas insuficientes ({n_bands} < 10) para índices espectrais")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # ========================================
    # CALCULAR ÍNDICES ESPECTRAIS
    # ========================================

    features_dict = {}

    # Definir regiões espectrais (índices aproximados)
    # Nota: Adaptar conforme a resolução espectral real do sensor
    
    # Assumindo ~256 bandas entre 1000-2500 nm → ~6 nm/banda
    # Região 1: ~1200 nm (lipídios CH2) → banda ~33
    # Região 2: ~1450 nm (água/proteínas OH, NH) → banda ~75
    # Região 3: ~1700 nm (lipídios CH) → banda ~117
    # Região 4: ~2100 nm (proteínas NH, CO) → banda ~183
    # Região 5: ~2300 nm (polissacarídeos OH) → banda ~217

    # Ajustar índices conforme número real de bandas
    idx_1200 = min(int(n_bands * 0.13), n_bands - 1)  # ~13% do espectro
    idx_1450 = min(int(n_bands * 0.30), n_bands - 1)  # ~30% do espectro
    idx_1700 = min(int(n_bands * 0.47), n_bands - 1)  # ~47% do espectro
    idx_2100 = min(int(n_bands * 0.73), n_bands - 1)  # ~73% do espectro
    idx_2300 = min(int(n_bands * 0.87), n_bands - 1)  # ~87% do espectro

    # Extrair bandas
    b_1200 = data[:, idx_1200]
    b_1450 = data[:, idx_1450]
    b_1700 = data[:, idx_1700]
    b_2100 = data[:, idx_2100]
    b_2300 = data[:, idx_2300]

    # 1. Normalized Difference Indices (NDI)
    features_dict['ndi_lipid_protein'] = (b_1200 - b_2100) / (b_1200 + b_2100 + 1e-8)
    features_dict['ndi_water_lipid'] = (b_1450 - b_1700) / (b_1450 + b_1700 + 1e-8)
    features_dict['ndi_protein_polysac'] = (b_2100 - b_2300) / (b_2100 + b_2300 + 1e-8)

    # 2. Simple Ratios (SR)
    features_dict['sr_lipid_protein'] = b_1200 / (b_2100 + 1e-8)
    features_dict['sr_water_lipid'] = b_1450 / (b_1700 + 1e-8)
    features_dict['sr_protein_polysac'] = b_2100 / (b_2300 + 1e-8)

    # 3. Profundidades de absorção relativas
    # (valor da banda / máximo do espectro)
    max_vals = np.max(data, axis=1, keepdims=True)
    features_dict['depth_1200'] = b_1200 / (max_vals.flatten() + 1e-8)
    features_dict['depth_1450'] = b_1450 / (max_vals.flatten() + 1e-8)
    features_dict['depth_1700'] = b_1700 / (max_vals.flatten() + 1e-8)
    features_dict['depth_2100'] = b_2100 / (max_vals.flatten() + 1e-8)
    features_dict['depth_2300'] = b_2300 / (max_vals.flatten() + 1e-8)

    # 4. Razões derivadas (combinações)
    features_dict['composite_lipid'] = (b_1200 + b_1700) / (2 * max_vals.flatten() + 1e-8)
    features_dict['composite_protein'] = (b_1450 + b_2100) / (2 * max_vals.flatten() + 1e-8)
    features_dict['composite_polysac'] = b_2300 / (max_vals.flatten() + 1e-8)

    # 5. Índices de posição
    features_dict['peak_position_normalized'] = np.argmax(data, axis=1) / n_bands

    # 6. Curvatura espectral em regiões-chave
    # Segunda derivada aproximada
    if n_bands >= 3:
        second_deriv = np.gradient(np.gradient(data, axis=1), axis=1)
        features_dict['curvature_1450'] = second_deriv[:, idx_1450]
        features_dict['curvature_2100'] = second_deriv[:, idx_2100]
    else:
        features_dict['curvature_1450'] = np.zeros(n_samples)
        features_dict['curvature_2100'] = np.zeros(n_samples)

    # Criar DataFrame
    features_df = pd.DataFrame(features_dict)

    # Salvar resultado
    features_df.to_csv(output_path, index=False)
    
    print(f"✅ Índices espectrais extraídos: {output_path}")
    print(f"   Dimensões: {features_df.shape[0]} amostras × {features_df.shape[1]} features")
    print(f"   Features: {list(features_df.columns)}")

    # Limpar arquivo intermediário
    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")