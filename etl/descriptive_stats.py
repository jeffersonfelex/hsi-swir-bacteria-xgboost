import os
import pandas as pd
import numpy as np

KEEP_INTERMEDIATE = False


def descriptive_stats(data: np.ndarray) -> pd.DataFrame:
    """
    Extrai estatísticas descritivas de cada espectro ao longo das bandas.

    Para cada amostra (linha), calcula:
        - mean     : média das bandas
        - std      : desvio padrão das bandas
        - min      : valor mínimo
        - max      : valor máximo
        - median   : mediana
        - skewness : assimetria
        - kurtosis : curtose

    Args:
        data: matriz [n_samples × n_bands]

    Returns:
        DataFrame [n_samples × 7] com prefixo 'desc_' nas colunas
    """
    if data.shape[0] == 0:
        raise ValueError("❌ descriptive_stats recebeu array vazio.")

    df_data = pd.DataFrame(data)

    return pd.DataFrame({
        "desc_mean":     np.mean(data, axis=1),
        "desc_std":      np.std(data, axis=1),
        "desc_min":      np.min(data, axis=1),
        "desc_max":      np.max(data, axis=1),
        "desc_median":   np.median(data, axis=1),
        "desc_skewness": df_data.skew(axis=1).values,
        "desc_kurtosis": df_data.kurtosis(axis=1).values,
    })


def run(output_dir: str, folder_name: str, suffix: str = "derivada"):
    """
    Lê um CSV de espectros, aplica descriptive_stats e salva o resultado.

    Args:
        output_dir: diretório onde estão os arquivos
        folder_name: nome base da imagem
        suffix: sufixo do arquivo de entrada (default: 'derivada')
    """
    input_path  = os.path.join(output_dir, f"{folder_name}_{suffix}.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"❌ Arquivo vazio: {input_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"❌ Nenhuma coluna numérica em {input_path}")

    stats_df = descriptive_stats(df[numeric_cols].values)
    stats_df.to_csv(output_path, index=False)

    print(f"✅ Features descritivas extraídas: {output_path}")
    print(f"   Dimensões : {stats_df.shape[0]} amostras × {stats_df.shape[1]} features")
    print(f"   Features  : {list(stats_df.columns)}")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Removido: {input_path}")