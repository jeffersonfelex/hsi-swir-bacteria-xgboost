import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

XGB_TOP_N = 50  # Top N features a selecionar

def run(input_csv_path: str, mi_features_path: str, output_dir: str):
    """
    Seleciona features usando XGBoost feature importance.
    Aplica sobre features pré-selecionadas por MI.
    
    Args:
        input_csv_path: Caminho para dataset_final.csv
        mi_features_path: Caminho para features_selected_mi.csv
        output_dir: Diretório para salvar resultados
    
    Returns:
        Lista com nomes das features selecionadas
    """
    print(f"📥 [XGBoost] Lendo dataset de {input_csv_path}")
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_csv_path}")
    
    if not os.path.exists(mi_features_path):
        raise FileNotFoundError(f"Arquivo MI não encontrado: {mi_features_path}")
    
    # Carregar dados
    df = pd.read_csv(input_csv_path)
    mi_features_df = pd.read_csv(mi_features_path)
    
    # Extrair features selecionadas por MI
    # Coluna pode ser 'band_name' ou 'feature_name'
    if 'feature_name' in mi_features_df.columns:
        mi_features = mi_features_df["feature_name"].tolist()
    elif 'band_name' in mi_features_df.columns:
        mi_features = mi_features_df["band_name"].tolist()
    else:
        raise ValueError("CSV de MI não tem coluna 'feature_name' ou 'band_name'")
    
    # Filtrar features válidas (que existem no dataset)
    valid_features = [f for f in mi_features if f in df.columns]
    
    if len(valid_features) == 0:
        raise ValueError("Nenhuma feature do MI encontrada no dataset!")
    
    print(f"   Features do MI disponíveis: {len(valid_features)}")
    
    # Preparar dados
    X = df[valid_features].fillna(0).values
    y = df["label"].values
    
    print(f"   Amostras: {len(y)} (positivas: {y.sum()}, negativas: {len(y) - y.sum()})")
    
    # Treinar XGBoost
    print(f"🌲 Treinando XGBoost...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    
    # Validação cruzada (opcional, para verificar performance)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Extrair feature importance
    importance = model.feature_importances_
    
    # Selecionar top N
    n_select = min(XGB_TOP_N, len(valid_features))
    idx_sorted = np.argsort(importance)[::-1][:n_select]
    
    top_features = np.array(valid_features)[idx_sorted]
    top_importance = importance[idx_sorted]
    
    # Salvar resultados
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features_selected_xgboost.csv")
    
    results_df = pd.DataFrame({
        "feature_name": top_features,
        "importance": top_importance,
        "rank": np.arange(1, len(top_features) + 1)
    })
    
    results_df.to_csv(output_path, index=False)
    
    print(f"✅ [XGBoost] {len(top_features)} features selecionadas")
    print(f"   Range importance: [{top_importance.min():.6f}, {top_importance.max():.6f}]")
    print(f"   Salvo em: {output_path}")
    
    return top_features.tolist()