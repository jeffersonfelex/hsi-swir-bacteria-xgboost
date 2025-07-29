import pandas as pd
import os
from xgboost import XGBClassifier

def feature_engineer(input_path: str, output_path: str, target_column: str = "classe", k: int = 20):
    """
    Realiza seleção de atributos com base na importância de ganho do XGBoost e salva o CSV reduzido.

    Parâmetros:
    - input_path: caminho do CSV com as features e a coluna alvo.
    - output_path: caminho para salvar o novo CSV com as features selecionadas.
    - target_column: nome da coluna-alvo (classe).
    - k: número de features mais importantes a serem selecionadas.
    """
    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    feature_gain = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.Series(feature_gain).sort_values(ascending=False)

    # XGBoost pode retornar nomes como 'f0', 'f1', ..., mapeie para os nomes reais do DataFrame
    col_map = {f"f{i}": col for i, col in enumerate(X.columns)}
    top_features = [col_map.get(f, f) for f in importance_df.head(k).index.tolist() if f in col_map]

    print(f"[XGBoost] Selecionadas {len(top_features)} bandas com maior ganho: {top_features}")

    df_selected = df[top_features + [target_column]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_selected.to_csv(output_path, index=False)
    print(f"Atributos selecionados salvos em: {output_path}")
