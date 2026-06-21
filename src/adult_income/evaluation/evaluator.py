import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


class ModelEvaluator:
    """
    Cross-validation, test-set evaluation, and comparison plots.

    Usage
    -----
    evaluator = ModelEvaluator(cv=5)

    # Cross-validation (before test)
    evaluator.cross_validate("xgboost", pipeline, X_train, y_train)

    # Final evaluation on held-out test set
    evaluator.evaluate_on_test("xgboost", pipeline, X_train, y_train, X_test, y_test)

    # Comparison table
    evaluator.comparison_table()
    """

    def __init__(self, cv: int = 5, random_state: int = 42) -> None:
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.cv_results: dict[str, dict] = {}
        self.test_results: dict[str, dict] = {}
        self._fitted_pipelines: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(self, name: str, pipeline, X, y) -> dict:
        """Run stratified CV and store mean ± std for each metric."""
        raw = cross_validate(pipeline, X, y, cv=self.cv, scoring=_SCORING, n_jobs=-1)
        summary = {
            metric: {
                "mean": raw[f"test_{metric}"].mean(),
                "std": raw[f"test_{metric}"].std(),
            }
            for metric in _SCORING
        }
        self.cv_results[name] = summary
        return summary

    def cross_validate_all(self, pipelines: dict, X, y) -> None:
        """Cross-validate a dict of {name: pipeline}."""
        for name, pipeline in pipelines.items():
            print(f"  Evaluating {name}...", end=" ", flush=True)
            self.cross_validate(name, pipeline, X, y)
            ap = self.cv_results[name]["average_precision"]["mean"]
            auc = self.cv_results[name]["roc_auc"]["mean"]
            print(f"AP={ap:.4f} AUC={auc:.4f}")

    # ------------------------------------------------------------------
    # Test-set evaluation
    # ------------------------------------------------------------------

    def evaluate_on_test(
        self, name: str, pipeline, X_train, y_train, X_test, y_test
    ) -> dict:
        """Fit on train, evaluate on test. Stores the fitted pipeline."""
        pipeline.fit(X_train, y_train)
        self._fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
        }
        self.test_results[name] = metrics
        return metrics

    def print_classification_report(self, name: str, pipeline, X_test, y_test) -> None:
        y_pred = pipeline.predict(X_test)
        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # Comparison tables
    # ------------------------------------------------------------------

    def cv_comparison_table(self, sort_by: str = "roc_auc") -> pd.DataFrame:
        rows = []
        for model_name, metrics in self.cv_results.items():
            row = {"model": model_name}
            for metric, values in metrics.items():
                row[f"{metric}_mean"] = round(values["mean"], 4)
                row[f"{metric}_std"] = round(values["std"], 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values(f"{sort_by}_mean", ascending=False).reset_index(drop=True)

    def test_comparison_table(self, sort_by: str = "roc_auc") -> pd.DataFrame:
        rows = [
            {"model": name, **metrics} for name, metrics in self.test_results.items()
        ]
        df = pd.DataFrame(rows)
        for col in df.columns[1:]:
            df[col] = df[col].round(4)
        return df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_cv_comparison(
        self, metric: str = "roc_auc", figsize: tuple = (12, 6)
    ) -> plt.Figure:
        df = self.cv_comparison_table(sort_by=metric)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            df["model"],
            df[f"{metric}_mean"],
            xerr=df[f"{metric}_std"],
            align="center",
            alpha=0.85,
            color="steelblue",
            ecolor="black",
            capsize=4,
        )
        ax.set_xlabel(metric)
        ax.set_title(f"CV model comparison — {metric} (mean ± std)")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def plot_confusion_matrix(
        self, name: str, X_test, y_test, figsize: tuple = (7, 6)
    ) -> plt.Figure:
        pipeline = self._fitted_pipelines[name]
        disp = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
        disp.figure_.set_size_inches(figsize)
        disp.ax_.set_title(f"Confusion matrix — {name}")
        return disp.figure_

    def plot_roc_curves(
        self, X_test, y_test, names: list[str] | None = None, figsize: tuple = (9, 7)
    ) -> plt.Figure:
        target_names = names or list(self._fitted_pipelines)
        fig, ax = plt.subplots(figsize=figsize)
        for name in target_names:
            if name in self._fitted_pipelines:
                RocCurveDisplay.from_estimator(
                    self._fitted_pipelines[name], X_test, y_test, ax=ax, name=name
                )
        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.50)")
        ax.set_title("ROC Curves — all models")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_pr_curves(
        self, X_test, y_test, names: list[str] | None = None, figsize: tuple = (9, 7)
    ) -> plt.Figure:
        target_names = names or list(self._fitted_pipelines)
        fig, ax = plt.subplots(figsize=figsize)
        for name in target_names:
            if name in self._fitted_pipelines:
                PrecisionRecallDisplay.from_estimator(
                    self._fitted_pipelines[name], X_test, y_test, ax=ax, name=name
                )
        baseline = (y_test == 1).mean()
        ax.axhline(
            baseline,
            color="k",
            linestyle="--",
            label=f"Random (AP={baseline:.2f})",
        )
        ax.set_title("Precision-Recall Curves — all models")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_metric_heatmap(self, figsize: tuple = (10, 6)) -> plt.Figure:
        df = self.test_comparison_table().set_index("model")
        fig, ax = plt.subplots(figsize=figsize)
        import seaborn as sns

        sns.heatmap(
            df.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlGn",
            ax=ax,
            linewidths=0.5,
            vmin=0.5,
            vmax=1.0,
        )
        ax.set_title("Test metrics heatmap")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Seuil de décision & métriques au seuil
    # ------------------------------------------------------------------

    @staticmethod
    def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
        """Métriques à un seuil donné (ROC-AUC / AP sont indépendants du seuil)."""
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        return {
            "seuil": round(float(threshold), 3),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "average_precision": average_precision_score(y_true, y_proba),
        }

    def threshold_comparison(self, y_true, y_proba, thresholds) -> pd.DataFrame:
        """Tableau des métriques pour plusieurs seuils (ex. 0,50 vs seuil retenu)."""
        return pd.DataFrame(
            [self.metrics_at_threshold(y_true, y_proba, t) for t in thresholds]
        )

    def plot_confusion_pair(
        self, y_true, y_pred, labels=("<=50K", ">50K"), figsize: tuple = (11, 4)
    ) -> plt.Figure:
        """Matrices de confusion effectifs + normalisée par ligne, côte à côte."""
        import seaborn as sns

        labels = list(labels)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, mat, fmt, title in [
            (axes[0], cm, "d", "Confusion (effectifs)"),
            (axes[1], cm_norm, ".2f", "Confusion (normalisée par ligne)"),
        ]:
            sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(title)
            ax.set_xlabel("Prédiction")
            ax.set_ylabel("Vraie classe")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Analyse d'erreurs & équité
    # ------------------------------------------------------------------

    @staticmethod
    def _error_type(yt, yp) -> str:
        if yt == 1 and yp == 1:
            return "Vrai positif"
        if yt == 0 and yp == 0:
            return "Vrai négatif"
        if yt == 0 and yp == 1:
            return "Faux positif"
        return "Faux négatif"

    def error_breakdown(self, X, y_true, y_pred, y_proba, profile_cols):
        """Table d'analyse d'erreurs + profils numériques moyens par type.

        Retourne ``(comptes, profils)`` et stocke la table enrichie dans
        ``self.error_df`` (réutilisée par ``group_metrics``).
        """
        df = X.copy()
        df["y_true"] = np.asarray(y_true)
        df["y_pred"] = np.asarray(y_pred)
        df["proba_>50K"] = np.asarray(y_proba)
        df["type_erreur"] = [
            self._error_type(t, p) for t, p in zip(df["y_true"], df["y_pred"])
        ]
        self.error_df = df
        counts = df["type_erreur"].value_counts()
        profiles = df.groupby("type_erreur")[list(profile_cols)].mean().round(2)
        return counts, profiles

    def group_metrics(self, group_col, df=None) -> pd.DataFrame:
        """Précision/rappel/F1 + FPR/FNR par modalité d'une variable (équité).

        Utilise ``self.error_df`` (issu de ``error_breakdown``) si ``df`` est None.
        """
        df = self.error_df if df is None else df
        rows = []
        for g, part in df.groupby(group_col):
            yt, yp = part["y_true"], part["y_pred"]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            rows.append({
                "groupe": g,
                "effectif": len(part),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
                "FNR": fn / (fn + tp) if (fn + tp) else np.nan,
            })
        return (
            pd.DataFrame(rows)
            .sort_values("effectif", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Importance des variables
    # ------------------------------------------------------------------

    def feature_importance(self, pipeline, top_n: int = 20):
        """Importances natives (arbres/boosting) sur les colonnes post-prétraitement.

        Retourne un DataFrame trié, ou ``None`` si le modèle n'expose pas
        ``feature_importances_``.
        """
        pre = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        if not hasattr(clf, "feature_importances_"):
            return None
        return (
            pd.DataFrame({
                "variable": pre.get_feature_names_out(),
                "importance": clf.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def plot_feature_importance(
        self, pipeline, top_n: int = 20, figsize: tuple = (8, 6)
    ) -> plt.Figure | None:
        """Barres horizontales des ``top_n`` variables les plus importantes."""
        imp = self.feature_importance(pipeline, top_n)
        if imp is None:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(imp["variable"][::-1], imp["importance"][::-1], color="steelblue")
        ax.set_title(f"{top_n} variables les plus importantes")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        return fig
