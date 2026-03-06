import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

MI_TOP_N = 100  # Top N features a selecionar


def feature_selection_mi(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Seleciona as top N features por Mutual Information.

    Etapa de filtragem inicial — reduz o espaço de features antes do XGBoost,
    mantendo apenas as com maior relevância estatística em relação ao target.

    Args:
        X            : matriz [n_samples × n_features]
        y            : vetor de labels [n_samples]
        feature_names: nomes das colunas de X

    Returns:
        X_selected    : matriz [n_samples × n_selected]
        selected_names: lista com nomes das features selecionadas
    """
    if len(np.unique(y)) < 2:
        raise ValueError("❌ Labels sem variabilidade — todas as amostras têm o mesmo rótulo.")

    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)

    n_select = min(MI_TOP_N, len(feature_names))
    idx_sorted = np.argsort(mi_scores)[::-1][:n_select]

    selected_names = [feature_names[i] for i in idx_sorted]
    X_selected     = X[:, idx_sorted]

    print(f"  📊 MI — {len(feature_names)} → {len(selected_names)} features")
    print(f"     Range MI: [{mi_scores[idx_sorted].min():.6f}, {mi_scores[idx_sorted].max():.6f}]")

    return X_selected, selected_names


def run(input_csv_path: str, output_dir: str) -> list[str]:
    """
    Lê um dataset CSV, aplica feature_selection_mi e salva o ranking.

    Args:
        input_csv_path: caminho para o dataset (ex: dataset_msc.csv)
        output_dir    : diretório para salvar features_selected_mi.csv

    Returns:
        lista com nomes das features selecionadas
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    if df.empty:
        raise ValueError("❌ Dataset vazio.")

    if "label" not in df.columns:
        raise ValueError("❌ Coluna 'label' não encontrada.")

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
    if not feature_cols:
        raise ValueError("❌ Nenhuma feature numérica encontrada.")

    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    print(f"📥 [MI] {len(feature_cols)} features | {len(y)} amostras")

    X_selected, selected_names = feature_selection_mi(X, y, feature_cols)

    # Salvar ranking
    os.makedirs(output_dir, exist_ok=True)
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
    idx_sorted = np.argsort(mi_scores)[::-1][:len(selected_names)]

    pd.DataFrame({
        "feature_name": selected_names,
        "mi_score":     mi_scores[idx_sorted],
        "rank":         np.arange(1, len(selected_names) + 1),
    }).to_csv(os.path.join(output_dir, "features_selected_mi.csv"), index=False)

    print(f"✅ [MI] Salvo em: {output_dir}/features_selected_mi.csv")
    return selected_names