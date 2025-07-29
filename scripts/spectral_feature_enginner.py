import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import os

def feature_importance_xgboost(df: pd.DataFrame, target_column: str = "classe"):
    X = df.drop(columns=[target_column])
    stats = {
        'mean': X.mean(axis=1),
        'std': X.std(axis=1),
        'kurtosis': kurtosis(X, axis=1),
        'skewness': skew(X, axis=1)
    }
    stats_df = pd.DataFrame(stats)
    stats_df[target_column] = df[target_column].values
    return stats_df

def derivadas_locais(df: pd.DataFrame, target_column: str = "classe"):
    X = df.drop(columns=[target_column])
    derivada_1 = X.diff(axis=1).iloc[:, 1:]
    derivada_2 = derivada_1.diff(axis=1).iloc[:, 1:]
    derivada_1.columns = [f'd1_{c}' for c in derivada_1.columns]
    derivada_2.columns = [f'd2_{c}' for c in derivada_2.columns]
    result = pd.concat([derivada_1, derivada_2], axis=1)
    result[target_column] = df[target_column].values
    return result

def razao_bandas(df: pd.DataFrame, bandas: list, target_column: str = "classe"):
    X = df.drop(columns=[target_column])
    razoes = {}
    for i in range(len(bandas)):
        for j in range(i + 1, len(bandas)):
            b1, b2 = bandas[i], bandas[j]
            if b1 in X.columns and b2 in X.columns:
                col_name = f'{b1}_div_{b2}'
                razoes[col_name] = X[b1] / (X[b2] + 1e-6)
    razoes_df = pd.DataFrame(razoes)
    razoes_df[target_column] = df[target_column].values
    return razoes_df

def ndvi_like(df: pd.DataFrame, nir: str, red: str, target_column: str = "classe"):
    X = df.drop(columns=[target_column])
    ndvi = (X[nir] - X[red]) / (X[nir] + X[red] + 1e-6)
    result = pd.DataFrame({f'ndvi_{nir}_{red}': ndvi})
    result[target_column] = df[target_column].values
    return result

def aplicar_engineering_e_salvar(input_path: str, output_path: str, bandas_razao: list,
                                 nir_band: str, red_band: str, target_column: str = "classe"):
    """
    Executa pipeline de engenharia de atributos e salva em CSV.
    """
    df = pd.read_csv(input_path)

    df_stats = feature_importance_xgboost(df, target_column)
    df_deriv = derivadas_locais(df, target_column)
    df_razao = razao_bandas(df, bandas_razao, target_column)
    df_ndvi = ndvi_like(df, nir_band, red_band, target_column)

    df_final = pd.concat([df_stats, df_deriv, df_razao, df_ndvi], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"[Engenharia] Dados salvos com atributos extraídos em: {output_path}")
