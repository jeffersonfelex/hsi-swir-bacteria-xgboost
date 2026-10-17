"""Testes que codificam as garantias de validacao descritas no guia de estudo/slides."""
import numpy as np
import pandas as pd
import pytest

from etl.derivative_transform import derivative_transform, WINDOW_LENGTH, POLYORDER
from etl.feature_selection_mi import feature_selection_mi
from etl.label_by_atcc import label_by_atcc
from etl.msc_signal import msc_signal
from ml import split_dataset
from ml.train import SCORING_CV, ic95, n_splits_para


# ---------------------------------------------------------------- ETL
def test_savgol_parametros_do_plano_de_trabalho():
    assert (WINDOW_LENGTH, POLYORDER) == (15, 2)


def test_derivada_de_rampa_e_constante():
    x = np.tile(np.linspace(0, 1, 64), (5, 1)) * 3.0
    d = derivative_transform(x)
    assert d.shape == x.shape
    assert np.allclose(d, d[0, 0], atol=1e-6)


def test_msc_remove_efeito_aditivo_e_multiplicativo():
    rng = np.random.default_rng(0)
    base = np.sin(np.linspace(0, 6, 128)) + 2
    data = np.stack([a + b * base for a, b in zip(rng.normal(0, .3, 20), rng.uniform(.5, 2, 20))])
    out = msc_signal(data)
    assert out.shape == data.shape
    assert out.std(axis=0).max() < data.std(axis=0).max() * 0.1


def test_rotulagem_atcc_e_resistente():
    df = pd.DataFrame({"sample_id": ["ATCC13_x", "ATCC16_y", "ATCC27_z", "503B_a", "426_1_b"]})
    assert label_by_atcc(df)["label"].tolist() == [1, 1, 1, 0, 0]


def test_selecao_mi_nao_filtra_quando_top_n_excede_n_features():
    """Caracteriza o achado: com 33 features e MI_TOP_N=100, MI mantem tudo
    (por isso os datasets _mi e _xgb dos branches msc/log/deriv sao identicos)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 33))
    y = (X[:, 0] > 0).astype(int)
    _, nomes = feature_selection_mi(X, y, [f"f{i}" for i in range(33)])
    assert len(nomes) == 33


# ---------------------------------------------------------------- split
def test_split_estratificado_por_imagem_deixa_resistente_nos_dois_lados():
    labels = pd.DataFrame({
        "sample_id": [f"clin{i}" for i in range(16)] + ["ATCC13", "ATCC16", "ATCC27"],
        "label": [0] * 16 + [1] * 3,
    })
    teste = split_dataset._test_images_por_classe(labels)
    res = labels.assign(teste=labels.sample_id.isin(teste)).groupby(["label", "teste"]).size()
    for classe in (0, 1):
        assert res[(classe, True)] >= 1 and res[(classe, False)] >= 1
    assert teste == split_dataset._test_images_por_classe(labels)  # deterministico


# ---------------------------------------------------------------- CV / metricas
def test_n_splits_adaptativo_limitado_pelas_imagens_resistentes():
    grupos = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["r1"] * 4 + ["r2"] * 4)
    y = np.array([0] * 16 + [1] * 8)
    assert n_splits_para(y, grupos) == 2


def test_ic95_com_poucos_folds_e_indefinido():
    lo, hi = ic95([0.1, 0.3])
    assert np.isnan(lo) and np.isnan(hi)


def test_ic95_limitado_a_0_1():
    lo, hi = ic95([0.0, 0.0, 1.0])
    assert 0.0 <= lo <= hi <= 1.0


class _Const:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (len(X), 1))


@pytest.mark.parametrize("metrica", list(SCORING_CV))
def test_fold_sem_positivos_da_nan_e_nao_zero(metrica):
    X = np.zeros((10, 2))
    assert np.isnan(SCORING_CV[metrica](_Const(), X, np.zeros(10, dtype=int)))


def test_recall_definido_quando_ha_positivos():
    X = np.zeros((10, 2))
    y = np.array([1] * 2 + [0] * 8)
    assert SCORING_CV["recall"](_Const(), X, y) == 0.0
