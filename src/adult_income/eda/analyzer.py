import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data.schemas import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET


class EDAAnalyzer:
    """Exploratory data analysis for the Adult Income dataset."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X.copy()
        self.y = y.copy()
        self.df = X.copy()
        self.df[TARGET] = y.values

    # ------------------------------------------------------------------
    # Textual summaries
    # ------------------------------------------------------------------

    def shape(self) -> tuple[int, int]:
        return self.X.shape

    def missing_values(self) -> pd.DataFrame:
        missing = self.X.isnull().sum()
        pct = missing / len(self.X) * 100
        df = pd.DataFrame({"count": missing, "pct": pct})
        return df[df["count"] > 0].sort_values("count", ascending=False)

    def class_balance(self) -> pd.DataFrame:
        counts = self.y.value_counts()
        return pd.DataFrame({"count": counts, "pct": counts / len(self.y) * 100})

    def numerical_stats(self) -> pd.DataFrame:
        return self.X[NUMERICAL_FEATURES].describe().T

    def categorical_stats(self) -> pd.DataFrame:
        rows = []
        for col in CATEGORICAL_FEATURES:
            rows.append(
                {
                    "feature": col,
                    "n_unique": self.X[col].nunique(),
                    "top": self.X[col].mode().iloc[0],
                    "missing": self.X[col].isnull().sum(),
                }
            )
        return pd.DataFrame(rows).set_index("feature")

    def summary(self) -> None:
        print(f"Dataset shape : {self.shape()}")
        print(f"\nClass balance :\n{self.class_balance()}\n")
        mv = self.missing_values()
        if mv.empty:
            print("No missing values detected.")
        else:
            print(f"Missing values :\n{mv}\n")
        print(f"\nNumerical statistics :\n{self.numerical_stats()}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_class_balance(self, figsize: tuple = (6, 4)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        cb = self.class_balance()
        ax.bar(cb.index, cb["pct"], color=["steelblue", "tomato"])
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Class balance")
        for i, (idx, row) in enumerate(cb.iterrows()):
            ax.text(i, row["pct"] + 0.5, f"{row['pct']:.1f}%", ha="center")
        fig.tight_layout()
        return fig

    def plot_numerical_distributions(self, figsize: tuple = (16, 10)) -> plt.Figure:
        n_cols = 3
        n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        palette = {"<=50K": "steelblue", ">50K": "tomato"}
        for ax, col in zip(axes.flat, NUMERICAL_FEATURES):
            for label, color in palette.items():
                mask = self.y == label
                ax.hist(self.X.loc[mask, col], alpha=0.6, label=label,
                        bins=30, color=color)
            ax.set_title(col)
            ax.legend(fontsize=8)
        for ax in axes.flat[len(NUMERICAL_FEATURES):]:
            ax.set_visible(False)
        fig.suptitle("Numerical feature distributions by income class", y=1.02)
        fig.tight_layout()
        return fig

    def plot_categorical_distributions(self, figsize: tuple = (20, 18)) -> plt.Figure:
        n_cols = 2
        n_rows = int(np.ceil(len(CATEGORICAL_FEATURES) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
            ct = pd.crosstab(self.X[col], self.y, normalize="index") * 100
            ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"],
                    rot=30, fontsize=8)
            ax.set_title(col)
            ax.set_ylabel("% within category")
            ax.legend(fontsize=7, loc="upper right")
        for ax in axes.flat[len(CATEGORICAL_FEATURES):]:
            ax.set_visible(False)
        fig.suptitle("Income rate per category", y=1.02)
        fig.tight_layout()
        return fig

    def plot_correlation_matrix(self, figsize: tuple = (10, 8)) -> plt.Figure:
        corr = self.X[NUMERICAL_FEATURES].corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5,
            mask=np.triu(np.ones_like(corr, dtype=bool)),
        )
        ax.set_title("Numerical feature correlations")
        fig.tight_layout()
        return fig

    def plot_missing_values(self, figsize: tuple = (8, 4)) -> plt.Figure:
        mv = self.missing_values()
        if mv.empty:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(mv.index, mv["pct"])
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing values per feature")
        fig.tight_layout()
        return fig
