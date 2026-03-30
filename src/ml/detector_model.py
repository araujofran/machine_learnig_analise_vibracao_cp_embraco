from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest


class EmbracoAnomalyDetector(BaseEstimator, OutlierMixin):
    """
    Encapsula toda a lógica de normalização (z-score em relação ao saudável),
    score do Isolation Forest, e Threshold Híbrido baseado no estado operacional.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        contamination: str | float = "auto",
        random_state: int = 42,
        hybrid_threshold_quantile: float = 0.95,
        target_operating_state: str = "steady_state",
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.hybrid_threshold_quantile = hybrid_threshold_quantile
        self.target_operating_state = target_operating_state

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.baseline_mean_: pd.Series | None = None
        self.baseline_std_: pd.Series | None = None
        self.hybrid_threshold_: float | None = None

    def fit(self, X: pd.DataFrame, df_context: pd.DataFrame) -> EmbracoAnomalyDetector:
        """
        Treina o Isolation Forest e calcula baselines usando X.
        df_context mapeia quais rótulos ('fault_mode', 'operating_state') as amostras de X possuem.
        Treinar com janelas (healthy/steady_state) é recomendado.
        """
        # Calcular baseline
        self.baseline_mean_ = X.mean()
        self.baseline_std_ = X.std().replace(0, 1e-6)

        # Treinar IF
        self.model.fit(X)

        # Obter scores
        normality_score = self.model.decision_function(X)

        X_norm = (X - self.baseline_mean_) / (self.baseline_std_ + 1e-6)
        X_norm = X_norm.clip(-5, 5)
        z_score_mean = X_norm.abs().mean(axis=1)

        hybrid_score = (-normality_score) + z_score_mean

        # Achar threshold
        healthy_steady_mask = (df_context["fault_mode"] == "healthy") & (
            df_context["operating_state"] == self.target_operating_state
        )

        if healthy_steady_mask.any():
            self.hybrid_threshold_ = float(
                hybrid_score[healthy_steady_mask].quantile(self.hybrid_threshold_quantile)
            )
        else:
            self.hybrid_threshold_ = float(hybrid_score.quantile(self.hybrid_threshold_quantile))

        return self

    def predict(self, X: pd.DataFrame, df_context: pd.DataFrame) -> pd.DataFrame:
        """
        Prediz (0=Normal, 1=Anomalia) usando limite híbrido.
        Filtra anomalias se o motor não estiver no estado-alvo (steady_state).
        Retorna as métricas e predições em um DataFrame copiado do context.
        """
        if self.baseline_mean_ is None or self.hybrid_threshold_ is None:
            raise ValueError("O modelo precisa ser treinado com .fit() antes.")

        df = df_context.copy()

        normality_score = self.model.decision_function(X)
        X_norm = (X - self.baseline_mean_) / (self.baseline_std_ + 1e-6)
        X_norm = X_norm.clip(-5, 5)
        z_score_mean = X_norm.abs().mean(axis=1).values

        hybrid_score = (-normality_score) + z_score_mean

        df["normality_score"] = normality_score
        df["z_score_mean"] = z_score_mean
        df["hybrid_score"] = hybrid_score
        df["hybrid_threshold"] = self.hybrid_threshold_

        # Predição Bruta (acima do hybrid_threshold)
        df["predicted_anomaly"] = (df["hybrid_score"] > self.hybrid_threshold_).astype(int)

        # Predição Operacional Filtrada (Uptime Filter)
        df["final_anomaly"] = 0
        mask = df["operating_state"] == self.target_operating_state
        df.loc[mask, "final_anomaly"] = df.loc[mask, "predicted_anomaly"]

        return df
