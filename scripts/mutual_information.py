import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif

def select_features_mi(input_path: str, output_path: str,
                                target_column: str = "classe", k: int = 20):
    """
    Seleciona as k bandas com maior Informação Mútua (MI) em relação à classe alvo e salva em CSV.

    Parâmetros:
    - input_path: caminho para o CSV com os dados.
    - output_path: caminho para salvar o CSV reduzido.
    - target_column: nome da coluna de classe.
    - k: número de atributos a manter.
    """
    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    top_features = mi_series.head(k).index.tolist()

    print(f"[MI] Selecionadas {len(top_features)} bandas com maior informação mútua: {top_features}")

    df_selected = df[top_features + [target_column]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_selected.to_csv(output_path, index=False)
    print(f"Atributos selecionados via MI salvos em: {output_path}")
if __name__ == "__main__":
    print("OK: módulo carregado")