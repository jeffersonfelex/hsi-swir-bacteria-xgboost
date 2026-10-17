"""
Divisao treino/teste POR IMAGEM (nunca por pixel), estratificada por classe.

Com apenas 3 isolados resistentes (ATCC13/16/27) um GroupShuffleSplit puro pode
deixar a classe resistente inteira no treino ou no teste. Aqui a divisao e feita
classe a classe, no nivel de imagem (sample_id), garantindo que:
  - cada classe tenha >= 1 imagem no treino E >= 1 imagem no teste;
  - pixels de uma mesma imagem nunca cruzem treino/teste (sem pseudo-replicacao).

A divisao e deterministica (ordenacao + embaralhamento com semente fixa).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import pandas as pd

BASE_DIR     = os.getenv("IC2025_BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")
OUTPUT_DIR   = os.path.join(BASE_DIR, "ml", "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PIPELINES = [
    "msc_mi", "msc_xgb",
    "log_mi", "log_xgb",
    "deriv_mi", "deriv_xgb",
    "raw_mi", "raw_xgb",
]

TEST_FRACTION = 0.30
RANDOM_STATE  = 42


def _test_images_por_classe(image_labels: pd.DataFrame) -> set:
    """Escolhe deterministicamente as imagens de teste, estratificando por classe."""
    rng = np.random.default_rng(RANDOM_STATE)
    test_images = set()

    for classe, grupo in image_labels.groupby("label"):
        ids = sorted(grupo["sample_id"].tolist())
        rng.shuffle(ids)  # embaralha de forma reprodutivel
        n = len(ids)

        # >= 1 imagem no teste, mas sempre deixando >= 1 no treino
        n_test = int(round(n * TEST_FRACTION))
        n_test = max(1, min(n_test, n - 1))

        test_images.update(ids[:n_test])
        print(f"    classe {classe}: {n} imagens -> {n_test} teste / {n - n_test} treino")

    return test_images


def split_by_image(pipeline_name: str) -> None:
    dataset_path = os.path.join(FEATURES_DIR, f"dataset_{pipeline_name}.csv")
    if not os.path.exists(dataset_path):
        print(f"  Dataset nao encontrado: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    if "sample_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset {pipeline_name} deve conter 'sample_id' e 'label'.")

    image_labels = df.groupby("sample_id")["label"].first().reset_index()

    print(f"\n  Pipeline: {pipeline_name}")
    print(f"  Total de imagens (grupos): {len(image_labels)}")
    dist = dict(zip(*np.unique(image_labels["label"], return_counts=True)))
    print(f"  Distribuicao de imagens por classe: {dist}")

    if image_labels["label"].nunique() < 2:
        raise ValueError(f"{pipeline_name}: apenas uma classe presente — impossivel dividir.")
    if (image_labels["label"].value_counts() < 2).any():
        print("  AVISO: alguma classe tem apenas 1 imagem no total; o teste ficara sem "
              "representante dessa classe ou o treino ficara sem ela.")

    test_images  = _test_images_por_classe(image_labels)
    train_images = set(image_labels["sample_id"]) - test_images

    df_train = df[df["sample_id"].isin(train_images)].reset_index(drop=True)
    df_test  = df[df["sample_id"].isin(test_images)].reset_index(drop=True)

    # Verificacoes de sanidade
    for nome, sub in (("treino", df_train), ("teste", df_test)):
        classes = set(np.unique(sub["label"]))
        assert classes == {0, 1}, (
            f"{pipeline_name}: conjunto de {nome} ficou sem as duas classes "
            f"(classes presentes: {classes}). Ajuste TEST_FRACTION/RANDOM_STATE."
        )

    train_path = os.path.join(OUTPUT_DIR, f"train_{pipeline_name}.csv")
    test_path  = os.path.join(OUTPUT_DIR, f"test_{pipeline_name}.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"  Treino: {len(df_train):>8} pixels | {len(train_images)} imagens | "
          f"classes {dict(zip(*np.unique(df_train['label'], return_counts=True)))}")
    print(f"  Teste : {len(df_test):>8} pixels | {len(test_images)} imagens | "
          f"classes {dict(zip(*np.unique(df_test['label'], return_counts=True)))}")
    print(f"  Imagens de teste: {sorted(test_images)}")
    print(f"  Salvo: {train_path}")
    print(f"  Salvo: {test_path}")


def main():
    print("Divisao treino/teste por imagem (estratificada por classe, sem vazamento)")
    print("=" * 70)
    for pipeline in PIPELINES:
        try:
            split_by_image(pipeline)
        except Exception as e:
            print(f"  Erro em {pipeline}: {e}")
    print("\nDivisao concluida.")


if __name__ == "__main__":
    main()
