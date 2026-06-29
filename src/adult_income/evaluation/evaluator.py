import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
)


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

    def out_of_fold_probabilities(self, pipeline, X, y, n_jobs: int = -1) -> np.ndarray:
        """Probabilités hors-pli de la classe positive (``cross_val_predict``).

        Chaque observation est notée par un modèle qui ne l'a pas vue en
        entraînement → estimation sans fuite, utilisée pour régler le seuil.
        """
        return cross_val_predict(
            pipeline, X, y, cv=self.cv, method="predict_proba", n_jobs=n_jobs
        )[:, 1]

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
        ax.set_title(f"CV model comparison : {metric} (mean ± std)")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def plot_confusion_matrix(
        self, name: str, X_test, y_test, figsize: tuple = (7, 6)
    ) -> plt.Figure:
        pipeline = self._fitted_pipelines[name]
        disp = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
        disp.figure_.set_size_inches(figsize)
        disp.ax_.set_title(f"Confusion matrix : {name}")
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
        ax.set_title("ROC Curves : all models")
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
        ax.set_title("Precision-Recall Curves : all models")
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

    @staticmethod
    def threshold_search_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
        """Précision / rappel / F1 pour une grille de seuils.

        ``thresholds`` par défaut = ``np.linspace(0.10, 0.90, 81)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.10, 0.90, 81)
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        rows = []
        for t in thresholds:
            y_hat = (y_proba >= t).astype(int)
            rows.append({
                "threshold": float(t),
                "precision": precision_score(y_true, y_hat, zero_division=0),
                "recall": recall_score(y_true, y_hat, zero_division=0),
                "f1": f1_score(y_true, y_hat, zero_division=0),
            })
        return pd.DataFrame(rows)

    @classmethod
    def find_best_threshold(
        cls, y_true, y_proba, metric: str = "f1", thresholds=None
    ) -> tuple[float, pd.DataFrame]:
        """Seuil maximisant ``metric`` (F1 par défaut) sur la grille.

        Retourne ``(seuil, table)``. En cas d'égalité, le **plus petit** seuil
        est retenu (tri secondaire déterministe).
        """
        table = cls.threshold_search_table(y_true, y_proba, thresholds)
        best_row = table.sort_values(
            [metric, "threshold"], ascending=[False, True]
        ).iloc[0]
        return float(best_row["threshold"]), table

    def plot_learning_curve(
        self,
        pipeline,
        X,
        y,
        scoring: str = "average_precision",
        train_sizes=None,
        model_name: str | None = None,
        figsize: tuple = (9, 5),
    ) -> plt.Figure:
        """Courbe d'apprentissage (entraînement vs validation croisée)."""
        if train_sizes is None:
            train_sizes = np.linspace(0.15, 1.0, 6)
        sizes, train_scores, val_scores = learning_curve(
            pipeline, X, y, cv=self.cv, scoring=scoring,
            train_sizes=train_sizes, n_jobs=-1,
        )
        train_mean, train_std = train_scores.mean(1), train_scores.std(1)
        val_mean, val_std = val_scores.mean(1), val_scores.std(1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sizes, train_mean, "o-", color="steelblue", label="Entraînement")
        ax.plot(sizes, val_mean, "o-", color="tomato", label="Validation croisée")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="steelblue")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="tomato")
        ax.set_xlabel("Taille de l'entraînement")
        ylabel = "Average Precision" if scoring == "average_precision" else scoring
        ax.set_ylabel(ylabel)
        title = "Courbe d'apprentissage"
        if model_name:
            title += f" : {model_name} optimisé"
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_threshold_curves(
        table: pd.DataFrame, best_threshold: float | None = None, figsize: tuple = (8, 4)
    ) -> plt.Figure:
        """Précision / rappel / F1 selon le seuil (sortie de ``threshold_search_table``)."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(table["threshold"], table["precision"], label="Précision")
        ax.plot(table["threshold"], table["recall"], label="Rappel")
        ax.plot(table["threshold"], table["f1"], label="F1", color="black", lw=2)
        if best_threshold is not None:
            ax.axvline(best_threshold, ls="--", color="gray",
                       label=f"Seuil retenu = {best_threshold:.2f}")
        ax.axvline(0.5, ls=":", color="lightgray", label="Seuil par défaut = 0,50")
        ax.set_xlabel("Seuil de décision")
        ax.set_ylabel("Score")
        ax.set_title("Compromis précision/rappel selon le seuil (prédictions hors-pli)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def class_weight_comparison(
        self, probes, X, y, threshold_metric: str = "f1"
    ) -> pd.DataFrame:
        """Effet de la pondération des classes, au seuil 0,50 ET au seuil réglé.

        Chaque sonde déclare **son** levier de pondération, car les modèles ne
        partagent pas le même paramètre (``class_weight`` pour les estimateurs
        sklearn, ``scale_pos_weight`` pour XGBoost ; le boosting sklearn n'a que
        ``sample_weight`` et n'est donc pas testable ici). Pour chaque sonde et
        chaque valeur du levier, on calcule des probabilités **hors-pli** puis les
        métriques au seuil par défaut (0,50) et au seuil maximisant
        ``threshold_metric`` (réglé sur les mêmes probas hors-pli).

        L'AP / ROC-AUC (classement) sont quasi insensibles à la pondération ;
        l'effet se voit sur F1/recall **à seuil fixe** : et la comparaison
        ``pondéré @0,50`` vs ``aucun @seuil*`` montre que la pondération fait
        surtout ce que le réglage du seuil fait déjà.

        Parameters
        ----------
        probes : list[dict]
            Chaque dict = ``{"name": str, "pipeline": Pipeline, "param": str,
            "values": dict[str, Any]}`` où ``values`` mappe un libellé vers la
            valeur du paramètre (ex. ``{"aucun": None, "balanced": "balanced"}``).
        """
        rows = []
        for spec in probes:
            for label, value in spec["values"].items():
                pipe = clone(spec["pipeline"])
                pipe.set_params(**{spec["param"]: value})
                proba = self.out_of_fold_probabilities(pipe, X, y)
                m_def = self.metrics_at_threshold(y, proba, 0.50)
                t_star, _ = self.find_best_threshold(y, proba, metric=threshold_metric)
                m_tun = self.metrics_at_threshold(y, proba, t_star)
                rows.append({
                    "modèle": spec["name"],
                    "pondération": label,
                    "AP": m_def["average_precision"],
                    "ROC_AUC": m_def["roc_auc"],
                    "F1@0.50": m_def["f1"],
                    "recall@0.50": m_def["recall"],
                    "seuil*": t_star,
                    "F1@seuil*": m_tun["f1"],
                    "recall@seuil*": m_tun["recall"],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def plot_grouped_bars(
        df: pd.DataFrame, title: str | None = None, ylabel: str | None = None,
        figsize: tuple = (9, 5),
    ) -> plt.Figure:
        """Barres groupées d'un DataFrame (index = groupes en X, colonnes = séries).

        L'axe Y démarre juste sous la plus petite valeur pour rendre les écarts
        lisibles.
        """
        ax = df.plot.bar(
            figsize=figsize, rot=20,
            color=["steelblue", "tomato", "seagreen", "goldenrod"][: df.shape[1]],
        )
        lo, hi = float(np.nanmin(df.to_numpy())), float(np.nanmax(df.to_numpy()))
        pad = (hi - lo) * 0.15 or 0.01
        ax.set_ylim(max(0.0, lo - 3 * pad), hi + pad)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig = ax.figure
        fig.tight_layout()
        return fig

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
