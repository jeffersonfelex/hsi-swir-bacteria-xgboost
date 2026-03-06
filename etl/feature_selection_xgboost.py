import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

XGB_TOP_N = 50  # Top N features a selecionar


def feature_selection_xgboost(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Refina a seleção de features usando XGBoost feature importance.

    Aplicado após MI — usa o ganho das árvores para selecionar as bandas
    com maior relevância para a classificação.

    A validação cruzada é feita com um estimator não treinado (sem data leakage)
    apenas para log de referência. O modelo de importância é treinado separadamente
    no conjunto completo, pois seu papel aqui é seleção, não avaliação final.

    Args:
        X            : matriz [n_samples × n_features] já filtrada pelo MI
        y            : vetor de labels [n_samples]
        feature_names: nomes das colunas de X

    Returns:
        X_selected    : matriz [n_samples × n_selected]
        selected_names: lista com nomes das features selecionadas
    """
    xgb_params = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    # Validação cruzada com estimator não treinado — apenas referência de performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(XGBClassifier(**xgb_params), X, y, cv=cv, scoring="accuracy")
    print(f"  📊 XGB CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Modelo treinado no conjunto completo para extração de importância
    model = XGBClassifier(**xgb_params)
    model.fit(X, y)

    importance = model.feature_importances_
    n_select   = min(XGB_TOP_N, len(feature_names))
    idx_sorted = np.argsort(importance)[::-1][:n_select]

    selected_names = [feature_names[i] for i in idx_sorted]
    X_selected     = X[:, idx_sorted]

    print(f"  📊 XGBoost — {len(feature_names)} → {len(selected_names)} features")
    print(f"     Range importance: [{importance[idx_sorted].min():.6f}, {importance[idx_sorted].max():.6f}]")

    return X_selected, selected_names


def run(input_csv_path: str, mi_features_path: str, output_dir: str) -> list[str]:
    """
    Lê o dataset e o ranking MI, aplica feature_selection_xgboost e salva.

    Args:
        input_csv_path  : caminho para o dataset (ex: dataset_msc.csv)
        mi_features_path: caminho para features_selected_mi.csv
        output_dir      : diretório para salvar features_selected_xgboost.csv

    Returns:
        lista com nomes das features selecionadas
    """
    for path in [input_csv_path, mi_features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Arquivo não encontrado: {path}")

    df         = pd.read_csv(input_csv_path)
    mi_df      = pd.read_csv(mi_features_path)
    mi_features = mi_df["feature_name"].tolist()

    valid_features = [f for f in mi_features if f in df.columns]
    if not valid_features:
        raise ValueError("❌ Nenhuma feature do MI encontrada no dataset.")

    X = df[valid_features].fillna(0).values
    y = df["label"].values

    print(f"📥 [XGBoost] {len(valid_features)} features | {len(y)} amostras")

    X_selected, selected_names = feature_selection_xgboost(X, y, valid_features)

    # Salvar ranking
    os.makedirs(output_dir, exist_ok=True)
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1, eval_metric="logloss")
    model.fit(X, y)
    importance = model.feature_importances_
    idx_sorted = np.argsort(importance)[::-1][:len(selected_names)]

    pd.DataFrame({
        "feature_name": selected_names,
        "importance":   importance[idx_sorted],
        "rank":         np.arange(1, len(selected_names) + 1),
    }).to_csv(os.path.join(output_dir, "features_selected_xgboost.csv"), index=False)

    print(f"✅ [XGBoost] Salvo em: {output_dir}/features_selected_xgboost.csv")
    return selected_names