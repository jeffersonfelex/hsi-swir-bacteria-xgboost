"""
Treina APENAS os pipelines RAW (baseline) e adiciona os resultados
ao comparacao_pipelines.csv existente, sem re-treinar os outros 6.

Uso: python ml/train_raw.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Adiciona raiz do projeto ao path para importar train.py
sys.path.insert(0, os.path.dirname(__file__))

from train import (
    treinar, detectar_dispositivo, formatar_tempo,
    RESULTS_DIR, TOTAL_FITS, TOTAL_COMBINACOES,
    _gerar_boxplot_comparativo,
)


RAW_PIPELINES = [
    ("raw", "mi"),
    ("raw", "xgb"),
]


def main():
    total    = len(RAW_PIPELINES)
    t_global = time.time()
    tempos   = []

    print("=" * 70)
    print("Treinamento — apenas pipelines RAW (baseline)")
    print("   Raw × MI  |  Raw × XGBoost Importance")
    print(f"\n   Pipeline: StandardScaler → PCA (95% var) → XGBoost")
    print(f"   Combinações por pipeline : {TOTAL_COMBINACOES} × 5 folds = {TOTAL_FITS}")
    print("=" * 70)

    device = detectar_dispositivo()

    resultados = []
    for idx, (branch, selecao) in enumerate(RAW_PIPELINES, start=1):
        t0      = time.time()
        result  = treinar(branch, selecao, idx, total, device)
        elapsed = time.time() - t0
        tempos.append(elapsed)
        resultados.append(result)
        print(f"\n  Tempo deste pipeline : {formatar_tempo(elapsed)}")

    # ── Atualizar comparacao_pipelines.csv ──
    comp_path = os.path.join(RESULTS_DIR, "comparacao_pipelines.csv")

    resultados_validos = [r for r in resultados if "cv_recall_mean" in r]
    if not resultados_validos:
        print("\nNenhum pipeline RAW concluído com sucesso.")
        return

    cols_resumo = [
        "pipeline", "branch", "selecao",
        "cv_recall_mean", "cv_recall_std", "cv_recall_ic95_lower", "cv_recall_ic95_upper",
        "cv_f1_mean", "cv_f1_std",
        "cv_auprc_mean", "cv_auprc_std",
        "folds_recall_zero",
        "insample_recall", "insample_f1",
        "n_features_originais", "n_componentes_pca", "n_grupos_imagens",
        "tempo_s",
    ]

    df_new = pd.DataFrame(resultados_validos)[cols_resumo]

    if os.path.exists(comp_path):
        df_existing = pd.read_csv(comp_path)
        # Remover raw antigos se existirem
        df_existing = df_existing[~df_existing["pipeline"].str.startswith("raw_")]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values("cv_recall_mean", ascending=False).reset_index(drop=True)
    df_combined.to_csv(comp_path, index=False)

    print("\n" + "=" * 70)
    print("COMPARAÇÃO ATUALIZADA — todos os pipelines")
    print("=" * 70)
    print(df_combined.to_string(index=False))

    print(f"\nTempo total RAW: {formatar_tempo(time.time() - t_global)}")
    print(f"Relatório atualizado: {comp_path}")

    # Regenerar boxplot comparativo com todos os pipelines
    all_results = df_combined.to_dict("records")
    _gerar_boxplot_comparativo(all_results)


if __name__ == "__main__":
    main()
