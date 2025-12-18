import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

MI_TOP_N = 100  # Top N features a selecionar

def run(input_csv_path: str, output_dir: str):
    """
    Seleciona features usando Mutual Information (MI).
    
    Args:
        input_csv_path: Caminho para dataset_final.csv
        output_dir: Diretório para salvar resultados
    
    Returns:
        Lista com nomes das features selecionadas
    """
    print(f"📥 [MI] Lendo dataset de {input_csv_path}")
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    
    if df.empty:
        raise ValueError("Dataset vazio!")
    
    # Verificar se coluna label existe
    if "label" not in df.columns:
        raise ValueError("Coluna 'label' não encontrada no dataset.")
    
    # Separar features e labels
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "label"]
    
    if len(feature_cols) == 0:
        raise ValueError("Nenhuma feature numérica encontrada!")
    
    # Tratar valores ausentes
    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    
    print(f"   Features totais: {len(feature_cols)}")
    print(f"   Amostras: {len(y)} (positivas: {y.sum()}, negativas: {len(y) - y.sum()})")
    
    # Verificar se há variabilidade nas labels
    if len(np.unique(y)) < 2:
        raise ValueError("Labels não têm variabilidade (todas iguais)!")
    
    # Calcular Mutual Information
    print(f"🧮 Calculando Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
    
    # Selecionar top N
    n_select = min(MI_TOP_N, len(feature_cols))
    idx_sorted = np.argsort(mi_scores)[::-1][:n_select]
    
    top_features = np.array(feature_cols)[idx_sorted]
    top_scores = mi_scores[idx_sorted]
    
    # Salvar resultados
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features_selected_mi.csv")
    
    results_df = pd.DataFrame({
        "feature_name": top_features,
        "mi_score": top_scores,
        "rank": np.arange(1, len(top_features) + 1)
    })
    
    results_df.to_csv(output_path, index=False)
    
    print(f"✅ [MI] {len(top_features)} features selecionadas")
    print(f"   Range MI: [{top_scores.min():.6f}, {top_scores.max():.6f}]")
    print(f"   Salvo em: {output_path}")
    
    return top_features.tolist()