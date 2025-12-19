import pandas as pd
import numpy as np

# Identificadores de amostras ATCC conhecidas (cepas resistentes de referência)
AMOSTRAS_ATCC = ["ATCC13", "ATCC16", "ATCC27"]


def gerar_labels_por_atcc(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Gera labels binárias baseadas na identificação das amostras ATCC.
    
    Labels (simples e direto):
    - ATCCs → label = 1 (resistente - referência conhecida)
    - Outras amostras → label = 0 (sensível)
    
    Esses labels são usados para seleção de bandas:
    1. Mutual Information (MI)
    2. XGBoost Feature Importance
    
    Args:
        df: DataFrame com features e coluna 'filename'
    
    Returns:
        DataFrame com coluna 'label' adicionada
    """
    
    print(f"\n{'='*70}")
    print("🏷️  GERAÇÃO DE LABELS (ATCC = Resistente)")
    print(f"{'='*70}")
    
    # Validações
    if df is None or df.empty:
        print("❌ DataFrame vazio!")
        df["label"] = 0
        return df
    
    if "filename" not in df.columns:
        print("❌ Coluna 'filename' não encontrada!")
        df["label"] = 0
        return df
    
    print(f"📊 Dataset: {len(df):,} linhas")
    print(f"📋 Amostras únicas: {df['filename'].nunique()}")
    
    # Identificar ATCCs
    pattern = '|'.join(AMOSTRAS_ATCC)
    atcc_mask = df["filename"].str.contains(pattern, case=False, na=False)
    
    n_atcc_linhas = atcc_mask.sum()
    atcc_filenames = df[atcc_mask]["filename"].unique()
    non_atcc_filenames = df[~atcc_mask]["filename"].unique()
    
    print(f"\n🔍 Padrões ATCC buscados: {AMOSTRAS_ATCC}")
    print(f"   ATCCs encontradas: {list(atcc_filenames)}")
    print(f"   Linhas ATCC: {n_atcc_linhas:,}")
    print(f"   Amostras clínicas: {list(non_atcc_filenames)}")
    
    if n_atcc_linhas == 0:
        print(f"\n⚠️ Nenhuma amostra ATCC encontrada!")
        df["label"] = 0
        return df
    
    # Gerar labels: ATCCs = 1, outras = 0
    df["label"] = atcc_mask.astype(int)
    
    n_resistentes = df["label"].sum()
    n_sensiveis = len(df) - n_resistentes
    
    print(f"\n🏷️ Labels gerados:")
    print(f"   Resistentes (ATCC):  {n_resistentes:,} ({n_resistentes/len(df)*100:.1f}%)")
    print(f"   Sensíveis (outros):  {n_sensiveis:,} ({n_sensiveis/len(df)*100:.1f}%)")
    
    # Resumo por amostra
    print(f"\n📋 Resumo por amostra:")
    resumo = df.groupby("filename").agg({"label": ["first", "count"]})
    resumo.columns = ["label", "n_pixels"]
    resumo["classe"] = resumo["label"].map({1: "RESISTENTE (ATCC)", 0: "SENSÍVEL"})
    print(resumo.to_string())
    
    print(f"\n{'='*70}")
    print("✅ Labels gerados! MI e XGBoost podem rodar.")
    print(f"{'='*70}\n")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python label_by_atcc.py <dataset.csv>")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    df = gerar_labels_por_atcc(df)
    
    output_path = sys.argv[1].replace(".csv", "_com_labels.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Salvo: {output_path}")