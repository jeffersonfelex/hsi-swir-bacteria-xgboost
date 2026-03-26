import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Caminhos
BASE_DIR     = r"C:\Users\Jefferson\Documents\GitHub\ic-2025"
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")
OUTPUT_DIR   = os.path.join(BASE_DIR, "ml", "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pipelines disponíveis
PIPELINES = [
    "msc_mi", "msc_xgb",
    "log_mi", "log_xgb",
    "deriv_mi", "deriv_xgb",
    "raw_mi", "raw_xgb",
]


def split_by_image(pipeline_name: str) -> None:
    """
    Divide o dataset em treino (70%) e teste (30%) por IMAGEM (sample_id),
    não por pixel. Isso garante que pixels da mesma imagem não apareçam
    simultaneamente no treino e no teste — evitando pseudo-replicação.

    A divisão é estratificada por label para manter a proporção de classes.
    """
    dataset_path = os.path.join(FEATURES_DIR, f"dataset_{pipeline_name}.csv")
    if not os.path.exists(dataset_path):
        print(f"  Dataset não encontrado: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    if "sample_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset {pipeline_name} deve conter 'sample_id' e 'label'.")

    # --- Agregar label por imagem (uma imagem = um label) ---
    image_labels = df.groupby("sample_id")["label"].first().reset_index()
    grupos = image_labels["sample_id"].values
    labels = image_labels["label"].values

    print(f"\n  Pipeline: {pipeline_name}")
    print(f"  Total de imagens (grupos): {len(grupos)}")
    print(f"  Distribuição de labels por imagem: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # GroupShuffleSplit garante que imagens inteiras ficam num único conjunto
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(grupos, labels, groups=grupos))

    train_images = set(grupos[train_idx])
    test_images  = set(grupos[test_idx])

    df_train = df[df["sample_id"].isin(train_images)].reset_index(drop=True)
    df_test  = df[df["sample_id"].isin(test_images)].reset_index(drop=True)

    # Salvar
    train_path = os.path.join(OUTPUT_DIR, f"train_{pipeline_name}.csv")
    test_path  = os.path.join(OUTPUT_DIR, f"test_{pipeline_name}.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path,  index=False)

    print(f"  Treino: {len(df_train)} pixels de {len(train_images)} imagens")
    print(f"  Teste : {len(df_test)} pixels de {len(test_images)} imagens")
    print(f"  Distribuição treino: {dict(zip(*np.unique(df_train['label'], return_counts=True)))}")
    print(f"  Distribuição teste : {dict(zip(*np.unique(df_test['label'], return_counts=True)))}")
    print(f"  Salvo: {train_path}")
    print(f"  Salvo: {test_path}")


def main():
    print("Divisão treino/teste por imagem (sem vazamento entre grupos)")
    print("=" * 60)

    for pipeline in PIPELINES:
        try:
            split_by_image(pipeline)
        except Exception as e:
            print(f"  Erro em {pipeline}: {e}")

    print("\nDivisão concluída para todos os pipelines.")


if __name__ == "__main__":
    main()
