import pandas as pd
import numpy as np

# Identificadores das cepas ATCC (resistentes de referência)
AMOSTRAS_ATCC = ["ATCC13", "ATCC16", "ATCC27"]


def label_by_atcc(df: pd.DataFrame, limiar_sim: float = None, **kwargs) -> pd.DataFrame:
    """
    Gera labels binárias baseadas na identificação das amostras ATCC.

    Labels:
        1 → resistente (cepas ATCC de referência)
        0 → sensível   (amostras clínicas)

    Args:
        df: DataFrame com coluna 'sample_id' ou 'filename'
        limiar_sim: Parâmetro opcional de similaridade

    Returns:
        DataFrame com coluna 'label' adicionada
    """
    print(f"\n{'='*60}")
    print("🏷️  Rotulagem — ATCC = Resistente")
    print(f"{'='*60}")

    if df is None or df.empty:
        print("❌ DataFrame vazio.")
        if df is not None:
            df["label"] = 0
        return df

    # Normalizar coluna identificadora
    col_id = None
    for cand in ["sample_id", "filename", "sample_name", "id"]:
        if cand in df.columns:
            col_id = cand
            break

    if col_id is None:
        raise ValueError("❌ Coluna identificadora ('sample_id' ou 'filename') não encontrada no DataFrame.")

    if "sample_id" not in df.columns:
        df["sample_id"] = df[col_id]
    if "filename" not in df.columns:
        df["filename"] = df[col_id]

    pattern = "|".join(AMOSTRAS_ATCC)
    atcc_mask = df["sample_id"].astype(str).str.contains(pattern, case=False, na=False)

    df["label"] = atcc_mask.astype(int)

    n_resistentes = int(df["label"].sum())
    n_sensiveis   = len(df) - n_resistentes

    print(f"  Amostras únicas  : {df['sample_id'].nunique()}")
    print(f"  Resistentes (1)  : {n_resistentes} ({n_resistentes/len(df)*100:.1f}%)")
    print(f"  Sensíveis   (0)  : {n_sensiveis}   ({n_sensiveis/len(df)*100:.1f}%)")

    if n_resistentes == 0:
        print("  ⚠️  Nenhuma amostra ATCC encontrada — todos os labels = 0.")

    return df


# Alias para compatibilidade
gerar_labels_por_atcc = label_by_atcc


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python label_by_atcc.py <dataset.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df = label_by_atcc(df)

    output_path = sys.argv[1].replace(".csv", "_com_labels.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Salvo: {output_path}")