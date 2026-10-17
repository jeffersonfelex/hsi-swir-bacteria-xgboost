"""
Selecao de atributos MI -> XGB-FI como etapa de um sklearn Pipeline.

A selecao e AJUSTADA somente com os dados que chegam em fit() (o treino, ou o
fold de treino durante a CV) e depois apenas APLICADA (colunas fixas) em
qualquer outro dado. Assim, nenhuma informacao de rotulo do teste externo (nem
do fold de validacao) participa da escolha das colunas.

  modo "mi"  : top MI_TOP_N atributos por Mutual Information
  modo "xgb" : MI (top MI_TOP_N) e, sobre esses, top XGB_TOP_N por importancia do XGBoost
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

from etl.feature_selection_mi import MI_TOP_N
from etl.feature_selection_xgboost import XGB_TOP_N

XGB_SELECTION_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=1, eval_metric="logloss",
)


class SelecaoMIXGB(BaseEstimator, TransformerMixin):
    """Transformer de selecao de atributos (ajustada so no que recebe em fit)."""

    def __init__(self, modo="mi", mi_top_n=MI_TOP_N, xgb_top_n=XGB_TOP_N,
                 max_amostras=50000, random_state=42):
        self.modo = modo
        self.mi_top_n = mi_top_n
        self.xgb_top_n = xgb_top_n
        self.max_amostras = max_amostras      # 0/None = usar todos os pixels
        self.random_state = random_state

    def fit(self, X, y):
        if self.modo not in ("mi", "xgb"):
            raise ValueError(f"modo invalido: {self.modo!r} (use 'mi' ou 'xgb')")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("Selecao: y com uma unica classe.")
        self.n_features_in_ = X.shape[1]

        # Subamostra (deterministica) so para acelerar MI/XGB; vem apenas de dados de treino.
        if self.max_amostras and len(y) > self.max_amostras:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(len(y), size=self.max_amostras, replace=False))
            if len(np.unique(y[idx])) < 2:
                idx = np.arange(len(y))
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y

        mi = mutual_info_classif(Xs, ys, random_state=self.random_state, n_neighbors=3)
        idx_mi = np.argsort(mi)[::-1][: min(self.mi_top_n, X.shape[1])]
        self.mi_scores_ = mi
        suporte = np.sort(idx_mi)

        if self.modo == "xgb":
            model = XGBClassifier(**XGB_SELECTION_PARAMS).fit(Xs[:, suporte], ys)
            imp = model.feature_importances_
            top = np.argsort(imp)[::-1][: min(self.xgb_top_n, len(suporte))]
            suporte = np.sort(suporte[top])

        self.support_ = suporte
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Esperado {self.n_features_in_} atributos, recebido {X.shape[1]}.")
        return X[:, self.support_]

    def nomes_selecionados(self, feature_names):
        return [feature_names[i] for i in self.support_]
