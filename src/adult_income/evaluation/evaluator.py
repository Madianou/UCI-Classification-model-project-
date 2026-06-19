import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
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
