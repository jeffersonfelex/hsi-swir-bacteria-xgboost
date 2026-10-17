"""Testes que codificam as garantias de validacao descritas no guia de estudo/slides."""
import numpy as np
import pandas as pd
import pytest

from etl.derivative_transform import derivative_transform, WINDOW_LENGTH, POLYORDER
from etl.feature_selection_mi import feature_selection_mi
from etl.label_by_atcc import label_by_atcc
from etl.msc_signal import msc_signal
from ml import split_dataset
from ml.selection import SelecaoMIXGB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


# ---------------------------------------------------------------- selecao so no treino
def _dados_selecao(n=300, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 3] + 0.5 * X[:, 7] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("modo,esperado", [("mi", 10), ("xgb", 5)])
def test_selecao_tamanho_e_aplicacao_no_teste(modo, esperado):
    X, y = _dados_selecao()
    sel = SelecaoMIXGB(modo=modo, mi_top_n=10, xgb_top_n=5).fit(X, y)
    assert len(sel.support_) == esperado
    assert {3, 7} & set(sel.support_)           # acha os atributos informativos
    assert sel.transform(X[:7]).shape == (7, esperado)


def test_selecao_no_pipeline_e_refeita_por_fold_e_ignora_validacao():
    """No Pipeline, fit() em dados diferentes escolhe colunas diferentes (reajuste por fold)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 20))
    y_a = (X[:, 2] > 0).astype(int)
    y_b = (X[:, 15] > 0).astype(int)
    pipe = Pipeline([("sel", SelecaoMIXGB(modo="mi", mi_top_n=1)), ("sc", StandardScaler())])
    pipe.fit(X, y_a)
    assert list(pipe.named_steps["sel"].support_) == [2]
    pipe.fit(X, y_b)
    assert list(pipe.named_steps["sel"].support_) == [15]


def test_selecao_rejeita_numero_de_atributos_diferente_na_transformacao():
    X, y = _dados_selecao()
    sel = SelecaoMIXGB(modo="mi", mi_top_n=5).fit(X, y)
    with pytest.raises(ValueError):
        sel.transform(X[:, :10])


def test_selecao_rejeita_uma_classe():
    X, _ = _dados_selecao()
    with pytest.raises(ValueError):
        SelecaoMIXGB().fit(X, np.zeros(len(X), dtype=int))


def test_selecao_subamostragem_deterministica():
    X, y = _dados_selecao(n=600)
    a = SelecaoMIXGB(modo="mi", mi_top_n=8, max_amostras=200).fit(X, y).support_
    b = SelecaoMIXGB(modo="mi", mi_top_n=8, max_amostras=200).fit(X, y).support_
    assert np.array_equal(a, b)


def test_main_nao_seleciona_atributos_antes_do_split():
    import ast, pathlib
    src = pathlib.Path(split_dataset.__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
    nomes = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    assert "feature_selection_mi" not in nomes and "feature_selection_xgboost" not in nomes
    assert not hasattr(__import__("ml.split_dataset", fromlist=["x"]), "feature_selection_mi")


def test_fluxo_split_treino_avaliacao_selecao_so_no_treino(tmp_path, monkeypatch):
    """Fim a fim (sintetico, minusculo): split -> treinar -> avaliar.
    A selecao e ajustada no treino; o teste so recebe as colunas escolhidas."""
    import joblib
    from ml import train, evaluate

    rng = np.random.default_rng(0)
    bandas = [f"b{i}" for i in range(30)]
    linhas = []
    imagens = [f"clin{i}" for i in range(8)] + ["ATCC13_a", "ATCC16_b", "ATCC27_c", "ATCC13_d"]
    for sid in imagens:
        y = int(sid.startswith("ATCC"))
        X = rng.normal(size=(40, 30))
        X[:, 4] += 3 * y                      # banda 4 e informativa
        d = pd.DataFrame(X, columns=bandas)
        d["sample_id"], d["label"] = sid, y
        linhas.append(d)
    feats_dir, data_dir = tmp_path / "features", tmp_path / "data"
    feats_dir.mkdir(); data_dir.mkdir()
    pd.concat(linhas).to_csv(feats_dir / "dataset_raw.csv", index=False)

    monkeypatch.setattr(split_dataset, "FEATURES_DIR", str(feats_dir))
    monkeypatch.setattr(split_dataset, "OUTPUT_DIR", str(data_dir))
    split_dataset.split_by_image("raw")
    assert (data_dir / "train_raw.csv").exists() and (data_dir / "test_raw.csv").exists()

    models, results, reports = tmp_path / "m", tmp_path / "r", tmp_path / "rep"
    for d in (models, results, reports):
        d.mkdir()
    monkeypatch.setattr(train, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(train, "FEATURES_DIR", str(feats_dir))
    monkeypatch.setattr(train, "MODEL_DIR", str(models))
    monkeypatch.setattr(train, "RESULTS_DIR", str(results))
    monkeypatch.setattr(train, "PARAM_GRID", {"xgb__n_estimators": [10], "xgb__max_depth": [2]})
    monkeypatch.setattr(train, "TOTAL_COMBINACOES", 1)
    monkeypatch.setattr(train, "SELECAO_LABEL", {"mi": "MI", "xgb": "XGB"})
    train.treinar("raw", "xgb", 1, 1, "cpu")

    pipe = joblib.load(models / "pipeline_raw_xgb.joblib")
    sel = pipe.named_steps["selecao"]
    assert sel.n_features_in_ == 30 and 4 in sel.support_
    assert len(sel.support_) <= 50

    monkeypatch.setattr(evaluate, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(evaluate, "MODEL_DIR", str(models))
    monkeypatch.setattr(evaluate, "REPORT_DIR", str(reports))
    m1 = evaluate.avaliar("raw_xgb")
    assert m1["n_pixels_teste"] > 0

    # Trocar os rotulos do teste NAO pode mudar as colunas (a avaliacao nao reajusta nada).
    suporte_antes = sel.support_.copy()
    df_te = pd.read_csv(data_dir / "test_raw.csv")
    df_te["label"] = 1 - df_te["label"]
    df_te.to_csv(data_dir / "test_raw.csv", index=False)
    evaluate.avaliar("raw_xgb")
    assert np.array_equal(joblib.load(models / "pipeline_raw_xgb.joblib").named_steps["selecao"].support_,
                          suporte_antes)
