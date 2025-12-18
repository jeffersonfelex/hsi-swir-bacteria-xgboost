import os
import pandas as pd
import numpy as np

KEEP_INTERMEDIATE = False  # ❌ Remove derivada após extrair features

def run(output_dir: str, folder_name: str):
    """
    Extrai features estatísticas descritivas de cada espectro.
    
    Para cada amostra (linha), calcula:
    - Média das bandas
    - Desvio padrão das bandas
    - Mínimo das bandas
    - Máximo das bandas
    - Mediana das bandas
    - Skewness (assimetria) das bandas
    - Kurtosis (curtose) das bandas
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo
    """
    # ✅ CORRETO: Ler arquivo após derivada
    input_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")

    # Verificar se arquivo existe
    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Pulando features descritivas.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Carregar dados
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Pulando features descritivas.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Extrair apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        print(f"⚠️ Nenhuma coluna numérica em {input_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df[numeric_cols].values
    n_samples = data.shape[0]

    # ✅ CORRETO: Calcular estatísticas POR AMOSTRA (ao longo das bandas)
    stats_dict = {
        'mean': np.mean(data, axis=1),           # Média de cada espectro
        'std': np.std(data, axis=1),             # Desvio padrão de cada espectro
        'min': np.min(data, axis=1),             # Mínimo de cada espectro
        'max': np.max(data, axis=1),             # Máximo de cada espectro
        'median': np.median(data, axis=1),       # Mediana de cada espectro
        'skewness': pd.DataFrame(data).skew(axis=1).values,    # ✅ Assimetria
        'kurtosis': pd.DataFrame(data).kurtosis(axis=1).values # ✅ Curtose
    }

    # Criar DataFrame com features descritivas
    # Formato: 1 linha por amostra, 1 coluna por estatística
    stats_df = pd.DataFrame(stats_dict)

    # Salvar resultado
    stats_df.to_csv(output_path, index=False)
    
    print(f"✅ Features descritivas extraídas: {output_path}")
    print(f"   Dimensões: {stats_df.shape[0]} amostras × {stats_df.shape[1]} features")
    print(f"   Features: {list(stats_df.columns)}")

    # Limpar arquivo intermediário
    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")