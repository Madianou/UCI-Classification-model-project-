import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..data.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    NUMERICAL_FEATURES,
    POSITIVE_CLASS,
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

    def numerical_stats(self, by_class: bool = False) -> pd.DataFrame:
        """describe() of the numerical features.

        by_class=True breaks the statistics down per income class (bivariate),
        so each feature's mean/median/std can be compared between <=50K and
        >50K — a quick read of which variables separate the classes.
        """
        if not by_class:
            return self.X[NUMERICAL_FEATURES].describe().T
        # rows = (feature, stat), columns = income class
        return self.df.groupby(TARGET)[NUMERICAL_FEATURES].describe().T

    def numeric_target_correlation(self) -> pd.Series:
        """Point-biserial correlation of each numerical feature with the target.

        Equivalent to Pearson between the feature and the 0/1-encoded income, so
        it captures the signed linear/monotonic link (positive = associated with
        >50K). Linear only: it underestimates non-linear features such as
        capital-gain — read it alongside the mutual information.
        """
        income_binary = (self.y == POSITIVE_CLASS).astype(int)
        return (
            self.X[NUMERICAL_FEATURES]
            .corrwith(income_binary)
            .sort_values(ascending=False)
        )

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

    @staticmethod
    def _cramers_v_from_chi2(
        chi2: float, n: int, r: int, k: int, bias_correction: bool = True
    ) -> float:
        """Cramér's V from a precomputed chi-square statistic and table shape.

        Shared by cramers_v (quali<->quali) and chi2_numeric_target (binned
        quanti<->target), so the Bergsma (2013) bias correction lives in one
        place.
        """
        if not bias_correction:
            denom = n * min(k - 1, r - 1)
            return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
        phi2 = chi2 / n
        phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        kcorr = k - (k - 1) ** 2 / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)
        return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0

    def cramers_v(self, col_a: str, col_b: str, bias_correction: bool = True) -> float:
        """Cramér's V association between two categorical columns, in [0, 1].

        0 = independent (every modality keeps the global split), 1 = perfect
        association. Derived from the chi-square of their crosstab; with
        bias_correction (Bergsma 2013) to avoid overestimation on small
        samples / high cardinality.
        """
        confusion = pd.crosstab(self.df[col_a], self.df[col_b])
        chi2 = stats.chi2_contingency(confusion)[0]
        n = confusion.to_numpy().sum()
        r, k = confusion.shape
        return self._cramers_v_from_chi2(chi2, n, r, k, bias_correction)

    def chi2_numeric_target(
        self,
        variables: str | list[str] | None = None,
        n_bins: int = 10,
        binning: str = "quantile",
        bias_correction: bool = True,
    ) -> pd.DataFrame:
        """Chi-square independence test between numerical feature(s) (binned)
        and the categorical target, with Cramér's V as the effect size.

        The chi-square test needs categories, so each quantitative variable is
        discretised first:
        - binning="quantile" -> equal-sized bins (pd.qcut), robust to skew, so
          every bin keeps enough observations for the chi-square expected-count
          assumption (the right default for fnlwgt, age, hours-per-week);
        - binning="uniform"  -> equal-width bins (pd.cut).

        At n~49k almost everything tests "significant", so always read p_value
        together with cramers_v: e.g. fnlwgt comes out significant (p~1e-7) yet
        with a negligible effect (Cramér's V~0.03) -> the drop decision holds.
        "Significatif != important".

        Returns one row per variable (chi2, p_value, dof, cramers_v, n_bins),
        sorted by effect size. NB: heavily zero-inflated features
        (capital-gain/loss) collapse to very few bins here -> prefer the
        has_capital_* binary split for those.
        """
        if variables is None:
            variables = NUMERICAL_FEATURES
        if isinstance(variables, str):
            variables = [variables]
        rows = []
        for col in variables:
            s = self.X[col]
            if binning == "quantile":
                binned = pd.qcut(s, q=n_bins, duplicates="drop")
            elif binning == "uniform":
                binned = pd.cut(s, bins=n_bins)
            else:
                raise ValueError("binning must be 'quantile' or 'uniform'")
            table = pd.crosstab(binned, self.y)
            chi2, p, dof, _ = stats.chi2_contingency(table)
            n = table.to_numpy().sum()
            r, k = table.shape
            rows.append(
                {
                    "feature": col,
                    "chi2": chi2,
                    "p_value": p,
                    "dof": dof,
                    "cramers_v": self._cramers_v_from_chi2(
                        chi2, n, r, k, bias_correction
                    ),
                    "n_bins": r,
                }
            )
        return (
            pd.DataFrame(rows)
            .set_index("feature")
            .sort_values("cramers_v", ascending=False)
        )

    def mann_whitney_target(
        self, variables: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Mann-Whitney U test of each numerical feature between the two income
        classes, with the rank-biserial correlation as effect size.

        Non-parametric (no normality assumption) and needs no binning, so it is
        the natural quanti<->binary-target test and avoids the artifacts of
        chi2_numeric_target (e.g. capital-gain/loss collapsing to a single bin).

        rank_biserial lies in [-1, 1]: the sign gives the direction (positive =
        feature larger among >50K), the magnitude the effect size (|r| < 0.1
        negligible). As always at n~49k, p_value is ~0 almost everywhere, so
        read it together with the effect size (e.g. fnlwgt: significant yet
        rank_biserial ~ 0 -> drop decision holds).

        Returns one row per variable (U, p_value, rank_biserial, median_pos,
        median_neg), sorted by absolute effect size.
        """
        if variables is None:
            variables = NUMERICAL_FEATURES
        if isinstance(variables, str):
            variables = [variables]
        is_pos = self.df[TARGET] == POSITIVE_CLASS
        rows = []
        for col in variables:
            pos = self.df.loc[is_pos, col].dropna()
            neg = self.df.loc[~is_pos, col].dropna()
            u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            # rank-biserial = 2 * P(pos > neg) - 1, derived from U
            rank_biserial = 2 * u / (len(pos) * len(neg)) - 1
            rows.append(
                {
                    "feature": col,
                    "U": u,
                    "p_value": p,
                    "rank_biserial": rank_biserial,
                    "median_pos": pos.median(),
                    "median_neg": neg.median(),
                }
            )
        out = pd.DataFrame(rows).set_index("feature")
        return out.reindex(out["rank_biserial"].abs().sort_values(ascending=False).index)

    def cramers_v_matrix(self, include_target: bool = False) -> pd.DataFrame:
        """Symmetric Cramér's V matrix between all categorical features.

        include_target=True adds the income target as a row/column, so its
        column ranks the quali<->target association (the categorical analog of
        numeric_target_correlation).
        """
        cols = list(CATEGORICAL_FEATURES)
        if include_target:
            cols = cols + [TARGET]
        mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                v = self.cramers_v(a, b)
                mat.loc[a, b] = mat.loc[b, a] = v
        return mat

    def income_rate_by(
        self,
        columns: str | list[str] | None = None,
        head: int | None = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """>50K rate and count per modality, for one or several categorical
        columns, sorted by rate (descending) within each variable.

        - columns=None  -> every categorical feature;
        - a single name -> a simple (modality-indexed) table;
        - several names -> one combined MultiIndex (variable x modality) table,
          to avoid printing many separate tables.
        - head=N keeps only the N highest and N lowest rates per variable,
          a compact view for high-cardinality variables (e.g. native-country).
        - dropna=False keeps missing values as a "(Manquant)" modality, whose
          rate can itself be informative (e.g. missing occupation -> mostly
          <=50K).
        """
        if columns is None:
            columns = CATEGORICAL_FEATURES
        if isinstance(columns, str):
            columns = [columns]
        parts = {}
        for col in columns:
            key = self.df[col] if dropna else self.df[col].fillna("(Manquant)")
            g = self.df.groupby(key)[TARGET]
            out = pd.DataFrame(
                {
                    "effectif": g.size(),
                    "taux_>50K (%)": g.apply(
                        lambda s: (s == POSITIVE_CLASS).mean() * 100
                    ),
                }
            ).sort_values("taux_>50K (%)", ascending=False)
            if head is not None and len(out) > 2 * head:
                out = pd.concat([out.head(head), out.tail(head)])
            out.index.name = "modalité"
            parts[col] = out
        if len(parts) == 1:
            return next(iter(parts.values()))
        return pd.concat(parts, names=["variable", "modalité"])

    def outlier_stats(self, whisker: float = 1.5) -> pd.DataFrame:
        """Per numerical feature, the IQR-rule outlier count, rate and bounds.

        A value is an outlier if it lies below Q1 - whisker*IQR or above
        Q3 + whisker*IQR (whisker=1.5 is the standard boxplot rule). Sorted by
        outlier rate, descending.
        """
        rows = []
        for col in NUMERICAL_FEATURES:
            s = self.X[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - whisker * iqr, q3 + whisker * iqr
            mask = (s < low) | (s > high)
            rows.append(
                {
                    "feature": col,
                    "n_outliers": int(mask.sum()),
                    "pct": mask.mean() * 100,
                    "lower": low,
                    "upper": high,
                }
            )
        return (
            pd.DataFrame(rows)
            .set_index("feature")
            .sort_values("pct", ascending=False)
        )

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

    def plot_numerical_bivariate(
        self,
        variable: str | None = None,
        filter_positive: bool = False,
        figsize: tuple | None = None,
    ) -> plt.Figure:
        """Bivariate view of numerical variable(s) vs the income class: one row
        per variable with the distribution (left) and a boxplot by class
        (right). Discrete variables get integer-aligned bins and no KDE.

        variable=None plots every numerical feature; pass a name to plot just
        one. filter_positive=True keeps only strictly positive values (useful
        for the zero-inflated capital features), applied per variable.
        """
        palette = {"<=50K": "steelblue", ">50K": "tomato"}
        variables = [variable] if variable is not None else NUMERICAL_FEATURES
        n = len(variables)
        figsize = figsize or (16, 4 * n)
        fig, axes = plt.subplots(n, 2, figsize=figsize, squeeze=False)
        for i, col in enumerate(variables):
            sub = self.df[self.df[col] > 0] if filter_positive else self.df
            s = sub[col].dropna()
            is_discrete = s.nunique() <= 25 and np.all(s == s.round())
            bins = np.arange(s.min() - 0.5, s.max() + 1.5, 1) if is_discrete else 30
            suffix = " (valeurs > 0)" if filter_positive else ""

            # Left: distribution split by income class
            sns.histplot(data=sub, x=col, hue=TARGET, bins=bins,
                         kde=not is_discrete, palette=palette, ax=axes[i, 0])
            axes[i, 0].set_title(f"Distribution de {col} selon le revenu{suffix}",
                                 fontweight="bold")
            axes[i, 0].set_ylabel("Nombre de personnes")

            # Right: boxplot per income class
            sns.boxplot(data=sub, x=TARGET, y=col, hue=TARGET,
                        palette=palette, legend=False, ax=axes[i, 1])
            axes[i, 1].set_title(f"Boxplot de {col} selon le revenu{suffix}",
                                 fontweight="bold")
            axes[i, 1].set_xlabel("Classe de revenu")

        fig.suptitle("Analyse des variables numériques selon la classe de revenu",
                     fontsize=18, fontweight="bold", y=1.005)
        fig.tight_layout()
        return fig

    def plot_numerical_univariate(
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

    def zero_proportion_by_class(
        self,
        variables: list[str] | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Bivariate: within each income class, zero vs non-zero breakdown for
        each variable (MultiIndex variable x income class).

        normalize=True returns row percentages (% within class); normalize=False
        returns raw counts. Designed for zero-inflated features (capital-gain,
        capital-loss): the gap between classes on the "> 0" column is where the
        discriminative signal lives.
        """
        variables = variables or ["capital-gain", "capital-loss"]
        parts = {}
        for var in variables:
            label = self.df[var].fillna(0).eq(0).map({True: "= 0", False: "> 0"})
            if normalize:
                ct = pd.crosstab(self.y, label, normalize="index") * 100
            else:
                ct = pd.crosstab(self.y, label)
            parts[var] = ct.reindex(columns=["= 0", "> 0"], fill_value=0)
        return pd.concat(parts.values(), keys=parts.keys(),
                         names=["variable", TARGET])

    def plot_zero_proportion_by_class(
        self,
        variables: list[str] | None = None,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Stacked bars of zero vs non-zero share per income class, one panel
        per variable (defaults to the zero-inflated capital features)."""
        variables = variables or ["capital-gain", "capital-loss"]
        table = self.zero_proportion_by_class(variables)
        fig, axes = plt.subplots(1, len(variables), figsize=figsize, squeeze=False)
        for ax, var in zip(axes.flat, variables):
            ct = table.loc[var]
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

    def plot_categorical_bivariate(self, figsize: tuple = (20, 18)) -> plt.Figure:
        n_cols = 2
        n_rows = int(np.ceil(len(CATEGORICAL_FEATURES) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Global >50K rate: a modality above this line over-represents high
        # income, below it under-represents it. Far from the line = strong signal.
        baseline = (self.y == POSITIVE_CLASS).mean() * 100
        for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
            ct = pd.crosstab(self.X[col], self.y, normalize="index") * 100
            ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"],
                    rot=30, fontsize=8)
            ax.axhline(baseline, ls="--", color="tomato", lw=1.2,
                       label=f"moy. {POSITIVE_CLASS} ({baseline:.1f}%)")
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

    def plot_cramers_v_matrix(
        self, include_target: bool = False, figsize: tuple = (9, 7)
    ) -> plt.Figure:
        """Heatmap of the Cramér's V matrix between categorical features
        (optionally including the target column)."""
        mat = self.cramers_v_matrix(include_target=include_target)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap="rocket_r",
            vmin=0, vmax=1, ax=ax, square=True, linewidths=0.5,
            mask=np.triu(np.ones_like(mat, dtype=bool), k=1),
        )
        ax.set_title("Cramér's V : associations entre variables catégorielles")
        fig.tight_layout()
        return fig

    def plot_categorical_crosstab(
        self,
        col_a: str,
        col_b: str,
        normalize: str | None = "index",
        figsize: tuple = (10, 7),
    ) -> plt.Figure:
        """Heatmap of the contingency table between two categorical features.

        normalize="index" -> row-conditional P(col_b | col_a) in %, the clearest
        way to read which modality of col_a maps to which of col_b (e.g. how
        marital-status maps onto relationship). normalize=None -> raw counts.
        Complements cramers_v: the V says how strong, this shows where.
        """
        norm = normalize or False
        ct = pd.crosstab(self.df[col_a], self.df[col_b], normalize=norm)
        if norm:
            ct = ct * 100
            fmt, cbar_label = ".0f", "% par ligne"
        else:
            fmt, cbar_label = "d", "effectif"
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(ct, annot=True, fmt=fmt, cmap="rocket_r", ax=ax,
                    linewidths=0.5, cbar_kws={"label": cbar_label})
        suffix = " (% par ligne)" if norm else " (effectifs)"
        ax.set_title(f"{col_a} × {col_b}{suffix}")
        fig.tight_layout()
        return fig

    def plot_education_equivalence(
        self,
        cat_col: str = "education",
        num_col: str = "education-num",
        figsize: tuple = (10, 5),
        verbose: bool = True,
    ) -> plt.Figure:
        """Overlay the >50K rate computed by cat_col (categorical) and by
        num_col (numeric), to prove the two encodings define identical groups.

        The curves coincide because each cat_col modality maps to exactly one
        num_col value. With verbose=True, also prints the two rate vectors and
        whether they are equal.
        """
        order = self.df.groupby(cat_col)[num_col].first().sort_values().index
        rc = (self.df.groupby(cat_col)[TARGET]
              .apply(lambda s: (s == POSITIVE_CLASS).mean() * 100).reindex(order))
        rn = (self.df.groupby(num_col)[TARGET]
              .apply(lambda s: (s == POSITIVE_CLASS).mean() * 100))
        identical = bool(np.allclose(rc.values, rn.values))

        if verbose:
            print(f"rc (par {cat_col})    :", rc.round(2).values)
            print(f"rn (par {num_col}):", rn.round(2).values)
            print("Taux identiques entre les deux groupements ?", identical)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rn.index, rc.values, "o-", color="steelblue", lw=2.5,
                label=f"par {cat_col} (catégoriel)")
        ax.plot(rn.index, rn.values, "x--", color="tomato", lw=1.2,
                label=f"par {num_col} (numérique)")
        for xi, yi, lab in zip(rn.index, rc.values, order):
            ax.annotate(lab, (xi, yi), rotation=45, fontsize=7,
                        ha="right", va="bottom")
        ax.set_xlabel(num_col)
        ax.set_ylabel("% >50K")
        ax.set_title(f"{cat_col} vs {num_col} — taux >50K superposés "
                     f"(identiques : {identical})")
        ax.legend(loc="upper left")
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
