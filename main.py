"""
Pipeline ETL — Imagens Hiperespectrais SWIR
===========================================
Classificacao espectral de bacterias multirresistentes com Gradient Boosting.

Fluxo (por imagem):
    ROI  ->  [ 4 branches EM PARALELO a partir do ROI ]
              raw   : espectro bruto (bandas)
              msc   : Multiplicative Signal Correction(ROI)
              log   : Log(1/R)(ROI)
              deriv : Savitzky-Golay 1a ordem(ROI)   (janela=15, grau=2)
     ->  engenharia de atributos (desc + spec + idx) POR BRANCH (msc/log/deriv)
     ->  raw usa as bandas diretamente
     ->  rotulagem por ATCC
     ->  selecao de features (MI e depois XGBoost) POR BRANCH
     ->  8 datasets em features_selected/: dataset_<branch>_<mi|xgb>.csv

Observacao: nas versoes anteriores os 4 branches eram encadeados
(msc -> log -> deriv) e apenas a derivada virava feature, o que tornava
MSC/Log/Deriv identicos. Agora cada branch parte do mesmo ROI.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import glob
import shutil
import logging
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etl import pre_processing_image
from etl.msc_signal import msc_signal
from etl.log_transform import log_transform
from etl.derivative_transform import derivative_transform
from etl.descriptive_stats import descriptive_stats
from etl.spectral_features import spectral_features
from etl.index_features import index_features
from etl.feature_selection_mi import feature_selection_mi
from etl.feature_selection_xgboost import feature_selection_xgboost
from etl.label_by_atcc import label_by_atcc


# ============================================================================
# CONFIGURACAO
# ============================================================================

@dataclass
class Config:
    BASE_DIR: str = os.getenv("IC2025_BASE_DIR", os.path.abspath(os.path.dirname(__file__)))

    @property
    def HSI_ROOT(self) -> str:
        return os.path.join(self.BASE_DIR, "dataSet", "hsi_original")

    @property
    def OUTPUT_BASE(self) -> str:
        return os.path.join(self.BASE_DIR, "dataSet", "data-preProcessed", "pre_processing")

    @property
    def FEATURES_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "features_selected")

    @property
    def LOG_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "logs")


config = Config()

ENGINEERED_BRANCHES = ["msc", "log", "deriv"]
ALL_BRANCHES = ENGINEERED_BRANCHES + ["raw"]

BRANCH_FN = {
    "msc":   msc_signal,
    "log":   log_transform,
    "deriv": derivative_transform,
}


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"pipeline_{ts}.log")

    logger = logging.getLogger("HSI_Pipeline")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log salvo em: {log_file}")
    return logger


logger = setup_logging()


# ============================================================================
# UTIL
# ============================================================================

def find_hdr_file(hsi_root: str, folder_name: str):
    candidates = [
        os.path.join(hsi_root, folder_name, "capture", f"{folder_name}.hdr"),
        os.path.join(hsi_root, folder_name, f"{folder_name}.hdr"),
        os.path.join(hsi_root, folder_name, "capture", "*.hdr"),
    ]
    for path in candidates:
        if "*" in path:
            hits = glob.glob(path)
            if hits:
                return hits[0]
        elif os.path.exists(path):
            return path
    return None


def _band_matrix(df: pd.DataFrame):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[cols].values.astype(np.float64), cols


def engineered_features(spectra: np.ndarray) -> pd.DataFrame:
    """desc_ (7) + spec_ (17) + idx_ (9) = 33 colunas para um branch."""
    parts = [
        descriptive_stats(spectra).reset_index(drop=True),
        spectral_features(spectra).reset_index(drop=True),
        index_features(spectra).reset_index(drop=True),
    ]
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


# ============================================================================
# FASE 1 — processamento por imagem
# ============================================================================

def process_single_sample(folder: str):
    logger.info(f"\n{'='*70}\nPROCESSANDO: {folder}\n{'='*70}")
    try:
        if not find_hdr_file(config.HSI_ROOT, folder):
            return False, "HDR nao encontrado"

        # ETAPA 1 — ROI + remocao de bandas ruidosas
        pre_processing_image.run(config.HSI_ROOT, folder)
        src_roi = os.path.join(config.HSI_ROOT, f"{folder}_roi_filtrado.csv")
        if not os.path.exists(src_roi):
            return False, "ROI nao gerado"

        roi_df = pd.read_csv(src_roi)
        if roi_df.empty or roi_df.shape[1] < 4:
            try:
                os.remove(src_roi)
            except OSError:
                pass
            return False, "ROI sem dados validos"

        # guarda o ROI em OUTPUT_BASE (para auditoria) e remove o intermediario
        dst_roi = os.path.join(config.OUTPUT_BASE, f"{folder}_roi.csv")
        roi_df.to_csv(dst_roi, index=False)
        try:
            os.remove(src_roi)
        except OSError:
            pass

        spectra, band_cols = _band_matrix(roi_df)
        n_pix, n_bands = spectra.shape
        logger.info(f"   ROI: {n_pix} pixels x {n_bands} bandas")

        if n_pix < 2:
            return False, "ROI com menos de 2 pixels"

        # ETAPA 2 — branches em paralelo a partir do ROI
        # raw = bandas do proprio ROI
        raw_df = pd.DataFrame(spectra.astype(np.float32), columns=band_cols)
        raw_df.insert(0, "sample_id", folder)
        raw_df.to_csv(os.path.join(config.OUTPUT_BASE, f"{folder}_raw_bands.csv"), index=False)

        # msc / log / deriv = engenharia de atributos sobre o espectro transformado
        for br in ENGINEERED_BRANCHES:
            transformed = BRANCH_FN[br](spectra)
            feats = engineered_features(np.asarray(transformed))
            feats.insert(0, "sample_id", folder)
            feats.to_csv(os.path.join(config.OUTPUT_BASE, f"{folder}_{br}_features.csv"), index=False)
            logger.info(f"   branch {br:5s}: {feats.shape[1]-1} features")

        logger.info(f"OK: {folder}")
        return True, "Sucesso"

    except Exception as e:
        logger.exception(f"Erro ao processar {folder}")
        return False, str(e)


# ============================================================================
# FASE 2 — combinar por branch
# ============================================================================

def combinar_branch(branch: str, folders: list) -> pd.DataFrame:
    if branch == "raw":
        dfs, band_sets = [], []
        for f in folders:
            p = os.path.join(config.OUTPUT_BASE, f"{f}_raw_bands.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            dfs.append(d)
            band_sets.append(set(d.columns) - {"sample_id"})
        if not dfs:
            raise FileNotFoundError("Nenhum *_raw_bands.csv encontrado.")
        common = sorted(set.intersection(*band_sets))
        logger.info(f"   raw: {len(common)} bandas comuns entre {len(dfs)} imagens")
        combined = pd.concat([d[["sample_id"] + common] for d in dfs], ignore_index=True)
    else:
        dfs = []
        for f in folders:
            p = os.path.join(config.OUTPUT_BASE, f"{f}_{branch}_features.csv")
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        if not dfs:
            raise FileNotFoundError(f"Nenhum *_{branch}_features.csv encontrado.")
        combined = pd.concat(dfs, ignore_index=True)

    combined = label_by_atcc(combined)
    # label_by_atcc adiciona 'filename' (duplicata de sample_id) — descartar
    drop = [c for c in ("filename",) if c in combined.columns]
    if drop:
        combined = combined.drop(columns=drop)
    return combined


# ============================================================================
# FASE 3 — selecao de features por branch
# ============================================================================

def selecionar_e_salvar(branch: str, df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    sid = df["sample_id"].values

    if len(np.unique(y)) < 2:
        logger.warning(f"   {branch}: labels sem variabilidade — selecao pulada.")
        return

    base = df.reset_index(drop=True)

    def _montar(nomes: list) -> pd.DataFrame:
        out = base[nomes].copy()
        out.insert(0, "sample_id", sid)
        out["label"] = y
        return out

    # MI
    _, names_mi = feature_selection_mi(X, y, feature_cols)
    path_mi = os.path.join(config.FEATURES_DIR, f"dataset_{branch}_mi.csv")
    _montar(names_mi).to_csv(path_mi, index=False)
    logger.info(f"   salvo: {os.path.basename(path_mi)}  ({len(names_mi)} features)")

    # XGBoost importance (sobre as features do MI)
    X_mi = base[names_mi].fillna(0).values
    _, names_xgb = feature_selection_xgboost(X_mi, y, names_mi)
    path_xgb = os.path.join(config.FEATURES_DIR, f"dataset_{branch}_xgb.csv")
    _montar(names_xgb).to_csv(path_xgb, index=False)
    logger.info(f"   salvo: {os.path.basename(path_xgb)}  ({len(names_xgb)} features)")

    pd.DataFrame({"feature": names_mi, "rank": range(1, len(names_mi) + 1)}).to_csv(
        os.path.join(config.FEATURES_DIR, f"features_selected_{branch}_mi.csv"), index=False)
    pd.DataFrame({"feature": names_xgb, "rank": range(1, len(names_xgb) + 1)}).to_csv(
        os.path.join(config.FEATURES_DIR, f"features_selected_{branch}_xgb.csv"), index=False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("INICIANDO PIPELINE ETL HSI")
    logger.info(f"   Base    : {config.BASE_DIR}")
    logger.info(f"   HSI Root: {config.HSI_ROOT}")

    os.makedirs(config.OUTPUT_BASE, exist_ok=True)
    os.makedirs(config.FEATURES_DIR, exist_ok=True)

    if not os.path.isdir(config.HSI_ROOT):
        logger.error(f"Diretorio HSI nao encontrado: {config.HSI_ROOT}")
        sys.exit(1)

    folders = sorted(
        f for f in os.listdir(config.HSI_ROOT)
        if os.path.isdir(os.path.join(config.HSI_ROOT, f))
    )
    if not folders:
        logger.error("Nenhuma pasta de amostra em hsi_original.")
        sys.exit(1)
    logger.info(f"Amostras encontradas: {len(folders)}")

    # ---- FASE 1 ----
    ok_folders, erros = [], []
    for folder in folders:
        success, msg = process_single_sample(folder)
        if success:
            ok_folders.append(folder)
        else:
            erros.append((folder, msg))
            logger.warning(f"   FALHOU {folder}: {msg}")

    logger.info(f"\n{'='*70}\nRESUMO FASE 1: {len(ok_folders)}/{len(folders)} imagens OK\n{'='*70}")
    for f, m in erros:
        logger.info(f"   - {f}: {m}")
    if not ok_folders:
        logger.error("Nenhuma imagem processada. Abortando.")
        sys.exit(1)

    # ---- FASE 2 + 3 ----
    for branch in ALL_BRANCHES:
        logger.info(f"\n{'='*70}\nBRANCH: {branch.upper()}\n{'='*70}")
        try:
            df = combinar_branch(branch, ok_folders)
        except Exception as e:
            logger.error(f"   combinacao de {branch} falhou: {e}")
            continue

        n_pos = int(df["label"].sum())
        logger.info(f"   {len(df)} pixels | resistentes {n_pos} | sensiveis {len(df) - n_pos} "
                    f"| imagens {df['sample_id'].nunique()}")

        # dataset combinado do branch (antes da selecao) — util para auditoria
        df.to_csv(os.path.join(config.FEATURES_DIR, f"dataset_{branch}.csv"), index=False)

        selecionar_e_salvar(branch, df)

    logger.info(f"\n{'='*70}\nPIPELINE ETL FINALIZADO\n{'='*70}")
    logger.info("Datasets em features_selected/: " +
                ", ".join(f"dataset_{b}_{s}.csv" for b in ALL_BRANCHES for s in ("mi", "xgb")))
    logger.info("Proximos passos: python ml/split_dataset.py -> python ml/train.py -> python ml/evaluate.py")
    logger.info(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
