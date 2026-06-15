import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..data.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    NUMERICAL_FEATURES,
    TARGET,
)


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

    def variable_overview(self) -> pd.DataFrame:
        """Per-column reference table: description, role, dtype, ML type, modalities."""
        rows = []
        for col in self.df.columns:
            is_numerical = col in NUMERICAL_FEATURES
            rows.append(
                {
                    "Variable": col,
                    "Description": FEATURE_DESCRIPTIONS.get(col, ""),
                    "Rôle": "Cible" if col == TARGET else "Variable explicative",
                    "Type pandas": str(self.df[col].dtype),
                    "Type ML": "Numérique" if is_numerical else "Catégorielle",
                    "Nombre de modalités": (
                        "" if is_numerical else self.df[col].nunique(dropna=True)
                    ),
                }
            )
        return pd.DataFrame(rows)

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

    def categorical_stats(self, rare_threshold: float = 1.0) -> pd.DataFrame:
        """Per categorical variable, a decision table for encoding choices.

        Adds to the basics (n_unique, top, missing):
        - top_pct  : share of the dominant modality (flags near-constant vars);
        - n_rare   : number of modalities below rare_threshold (%), candidates
                     to group into an "Other" bucket.
        """
        rows = []
        for col in CATEGORICAL_FEATURES:
            counts = self.X[col].value_counts()
            pct = counts / len(self.X) * 100
            rows.append(
                {
                    "feature": col,
                    "n_unique": self.X[col].nunique(),
                    "top": counts.index[0],
                    "top_pct": pct.iloc[0],
                    "n_rare": int((pct < rare_threshold).sum()),
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

    def plot_class_balance(self, figsize: tuple = (12, 4)) -> plt.Figure:
        cb = self.class_balance()
        colors = ["steelblue", "tomato"]
        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=figsize)

        # Left: raw counts
        ax_bar.bar(cb.index, cb["count"], color=colors)
        ax_bar.set_xlabel(TARGET)
        ax_bar.set_ylabel("count")
        for i, count in enumerate(cb["count"]):
            ax_bar.text(i, count, f"{count:,}", ha="center", va="bottom")

        # Right: percentage breakdown
        ax_pie.pie(
            cb["count"], labels=cb.index, colors=colors,
            autopct="%1.1f%%", startangle=90, counterclock=False,
        )
        ax_pie.axis("equal")

        fig.suptitle(f"Distribution de la variable cible : {TARGET.capitalize()}")
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

    def plot_numeric_diagnostics(
        self,
        variable: str,
        title: str | None = None,
        filter_positive: bool = False,
        bins: int = 30,
        kde: bool | None = None,
    ) -> plt.Figure:
        """Univariate diagnostics for one numerical variable, side by side:
        distribution (hist + KDE), boxplot, and normal QQ-plot.

        filter_positive=True keeps only strictly positive values, useful for the
        zero-inflated features (capital-gain, capital-loss) whose zero spike
        otherwise hides the shape of the actual values.
        """
        title = title or variable
        data = self.df[variable].dropna()
        if filter_positive:
            data = data[data > 0]

        # Discrete integer variables (e.g. education-num) need one bin per value,
        # otherwise fixed bins leave empty gaps between integers ("comb" effect).
        # The KDE also assumes continuity, so we drop it in that case.
        is_discrete = data.nunique() <= 25 and np.all(data == data.round())
        if is_discrete:
            bins_arg = np.arange(data.min() - 0.5, data.max() + 1.5, 1)
        else:
            bins_arg = bins

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # 1. Distribution
        sns.histplot(data, bins=bins_arg, kde=not is_discrete, ax=axes[0],
                     color="steelblue")
        axes[0].set_title(f"Distribution - {title}", fontweight="bold")
        axes[0].set_xlabel(title)
        axes[0].set_ylabel("Nombre de personnes")

        # 2. Boxplot
        sns.boxplot(x=data, ax=axes[1], color="steelblue")
        axes[1].set_title(f"Boxplot - {title}", fontweight="bold")
        axes[1].set_xlabel(title)

        # 3. QQ-plot (normality check)
        stats.probplot(data, dist="norm", plot=axes[2])
        axes[2].set_title(f"QQ-plot - {title}", fontweight="bold")

        suffix = " (valeurs > 0)" if filter_positive else ""
        fig.suptitle(f"Analyse univariée de {title}{suffix}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout()
        return fig

    def zero_proportion(self, variables: list[str] | None = None) -> pd.DataFrame:
        """Univariate: overall % of zero vs non-zero values per variable.

        For zero-inflated features (capital-gain, capital-loss), quantifies how
        dominant the zero spike is, without any split on the target.
        """
        variables = variables or ["capital-gain", "capital-loss"]
        rows = {}
        for var in variables:
            is_zero = self.df[var].fillna(0).eq(0)
            rows[var] = {"= 0": is_zero.mean() * 100, "> 0": (~is_zero).mean() * 100}
        return pd.DataFrame(rows).T[["= 0", "> 0"]]

    def plot_zero_proportion(
        self,
        variables: list[str] | None = None,
        figsize: tuple = (8, 4),
    ) -> plt.Figure:
        """Univariate: horizontal stacked bar of zero vs non-zero share per
        variable (no target split)."""
        prop = self.zero_proportion(variables)
        fig, ax = plt.subplots(figsize=figsize)
        prop.plot(kind="barh", stacked=True, ax=ax, color=["lightgrey", "tomato"])
        ax.set_xlabel("% of rows")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=8)
        for i, nonzero in enumerate(prop["> 0"]):
            ax.text(100 - nonzero / 2, i, f"{nonzero:.1f}%",
                    ha="center", va="center", fontweight="bold")
        ax.set_title("Proportion de valeurs nulles vs non nulles")
        fig.tight_layout()
        return fig

    def zero_proportion_by_class(self, variable: str) -> pd.DataFrame:
        """Within each income class, the % of rows that are zero vs non-zero.

        Designed for zero-inflated features (capital-gain, capital-loss): the
        gap between classes on the "> 0" column is where the discriminative
        signal lives (e.g. what share of >50K has a non-zero capital gain).
        """
        label = self.df[variable].fillna(0).eq(0).map({True: "= 0", False: "> 0"})
        ct = pd.crosstab(self.y, label, normalize="index") * 100
        return ct.reindex(columns=["= 0", "> 0"], fill_value=0)

    def plot_zero_proportion_by_class(
        self,
        variables: list[str] | None = None,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Stacked bars of zero vs non-zero share per income class, one panel
        per variable (defaults to the zero-inflated capital features)."""
        variables = variables or ["capital-gain", "capital-loss"]
        fig, axes = plt.subplots(1, len(variables), figsize=figsize, squeeze=False)
        for ax, var in zip(axes.flat, variables):
            ct = self.zero_proportion_by_class(var)
            ct.plot(kind="bar", stacked=True, ax=ax,
                    color=["lightgrey", "tomato"], rot=0)
            ax.set_title(var)
            ax.set_ylabel("% within class")
            ax.set_xlabel(TARGET)
            ax.legend(title=var, fontsize=8, loc="lower right")
            # Annotate the non-zero share (the discriminative part)
            for i, nonzero in enumerate(ct["> 0"]):
                ax.text(i, 100 - nonzero / 2, f"{nonzero:.1f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold")
        fig.suptitle("Proportion de valeurs nulles vs non nulles par classe", y=1.02)
        fig.tight_layout()
        return fig

    def plot_categorical_univariate(
        self,
        top_n: int = 10,
        figsize: tuple = (18, 22),
    ) -> plt.Figure:
        """Univariate distribution of each categorical variable.

        Horizontal sorted bars (best for long labels and dominant modalities).
        For high-cardinality variables (native-country, occupation, education),
        only the top_n modalities are shown and the rest are aggregated into an
        "Autres (k)" bar. Missing values appear as their own "(Manquant)" bar.
        Bars are annotated with the percentage of rows.
        """
        n_cols = 2
        n_rows = int(np.ceil(len(CATEGORICAL_FEATURES) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        total = len(self.X)
        for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
            vc = self.X[col].fillna("(Manquant)").value_counts()
            if len(vc) > top_n:
                n_grouped = len(vc) - top_n
                head = vc.iloc[:top_n].copy()
                head[f"Autres ({n_grouped})"] = vc.iloc[top_n:].sum()
                vc = head
            vc = vc.iloc[::-1]  # largest on top for horizontal bars
            ax.barh(vc.index.astype(str), vc.values, color="steelblue")
            for j, count in enumerate(vc.values):
                ax.text(count, j, f" {count / total * 100:.1f}%",
                        va="center", fontsize=8)
            ax.set_title(f"{col} ({self.X[col].nunique()} modalités)")
            ax.set_xlabel("count")
            ax.margins(x=0.12)  # room for the % labels
        for ax in axes.flat[len(CATEGORICAL_FEATURES):]:
            ax.set_visible(False)
        fig.suptitle("Distribution des variables catégorielles", y=1.005,
                     fontsize=14)
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
