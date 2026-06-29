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
    """Analyse exploratoire des données du dataset Adult Income."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X.copy()
        self.y = y.copy()
        self.df = X.copy()
        self.df[TARGET] = y.values

    # ------------------------------------------------------------------
    # Textual summaries
    # ------------------------------------------------------------------
    
    def shape(self) -> tuple[int, int]:
        """Dimensions (lignes, colonnes) des features."""
        return self.X.shape

    def variable_overview(self) -> pd.DataFrame:
        """Table de référence par colonne : description, rôle, dtype, type ML, modalités."""
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
        """Effectif et % de manquants par variable (seules celles qui en ont)."""
        missing = self.X.isnull().sum()
        pct = missing / len(self.X) * 100
        df = pd.DataFrame({"count": missing, "pct": pct})
        return df[df["count"] > 0].sort_values("count", ascending=False)

    def class_balance(self) -> pd.DataFrame:
        """Effectif et pourcentage de chaque classe de la cible."""
        counts = self.y.value_counts()
        return pd.DataFrame({"count": counts, "pct": counts / len(self.y) * 100})

    def numerical_stats(self, by_class: bool = False) -> pd.DataFrame:
        """describe() des variables numériques.

        by_class=True ventile les statistiques par classe de revenu (bivarié),
        pour comparer moyenne/médiane/écart-type de chaque variable entre <=50K
        et >50K : lecture rapide des variables qui séparent les classes.
        """
        if not by_class:
            return self.X[NUMERICAL_FEATURES].describe().T
        # rows = (feature, stat), columns = income class
        return self.df.groupby(TARGET)[NUMERICAL_FEATURES].describe().T

    def numeric_target_correlation(self) -> pd.Series:
        """Corrélation point-bisériale de chaque variable numérique avec la cible.

        Équivalente à un Pearson entre la variable et le revenu encodé 0/1 :
        capte le lien linéaire/monotone signé (positif = associé à >50K). Linéaire
        seulement : sous-estime les variables non linéaires comme capital-gain,
        à croiser avec les taux par modalité / distributions conditionnelles.
        """
        income_binary = (self.y == POSITIVE_CLASS).astype(int)
        return (
            self.X[NUMERICAL_FEATURES]
            .corrwith(income_binary)
            .sort_values(ascending=False)
        )

    def categorical_stats(self, rare_threshold: float = 1.0) -> pd.DataFrame:
        """Table d'aide à la décision d'encodage, par variable catégorielle.

        En plus des bases (n_unique, top, missing) :
        - top_pct : part de la modalité dominante (repère les variables quasi
                    constantes) ;
        - n_rare  : nombre de modalités sous rare_threshold (%), candidates à un
                    regroupement dans un bac "Autres".
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
        """Cramér's V à partir d'un chi² déjà calculé et de la forme du tableau.

        Partagé par cramers_v (quali<->quali) et chi2_numeric_target (quanti
        binné <->cible), pour que la correction de biais de Bergsma (2013) soit
        définie à un seul endroit.
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
        """Association de Cramér's V entre deux colonnes catégorielles, dans [0, 1].

        0 = indépendance (chaque modalité garde la répartition globale), 1 =
        association parfaite. Dérivée du chi² de leur tableau croisé ; avec
        bias_correction (Bergsma 2013) pour éviter la surestimation sur petits
        échantillons / forte cardinalité.
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
        """Test d'indépendance du chi² entre variable(s) numérique(s) (binnée(s))
        et la cible catégorielle, avec le Cramér's V comme taille d'effet.

        Le chi² a besoin de catégories, donc chaque variable quantitative est
        d'abord discrétisée :
        - binning="quantile" -> bacs de taille égale (pd.qcut), robuste à
          l'asymétrie, chaque bac garde assez d'observations pour l'hypothèse
          d'effectifs attendus du chi² (le bon défaut pour fnlwgt, age, hours) ;
        - binning="uniform"  -> bacs de largeur égale (pd.cut).

        À n~49k presque tout ressort "significatif" : toujours lire p_value avec
        cramers_v. Ex. fnlwgt est significatif (p~1e-7) mais d'effet négligeable
        (Cramér's V~0,03) -> la décision de drop tient. "Significatif != important".

        Renvoie une ligne par variable (chi2, p_value, dof, cramers_v, n_bins),
        triée par taille d'effet. NB : les variables très zero-inflated
        (capital-gain/loss) s'effondrent en très peu de bacs ici -> préférer pour
        elles l'indicateur binaire has_capital_*.
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
                raise ValueError("binning doit valoir 'quantile' ou 'uniform'")
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

    def cramers_v_matrix(self, include_target: bool = False) -> pd.DataFrame:
        """Matrice symétrique de Cramér's V entre toutes les catégorielles.

        include_target=True ajoute la cible income en ligne/colonne : sa colonne
        classe l'association quali<->cible (l'analogue catégoriel de
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
        """Taux de >50K et effectif par modalité, pour une ou plusieurs colonnes
        catégorielles, trié par taux (décroissant) au sein de chaque variable.

        - columns=None  -> toutes les variables catégorielles ;
        - un seul nom   -> une table simple (indexée par modalité) ;
        - plusieurs noms -> une table MultiIndex (variable x modalité) combinée,
          pour éviter d'imprimer beaucoup de tables séparées.
        - head=N ne garde que les N taux les plus hauts et N les plus bas par
          variable : vue compacte pour les variables à forte cardinalité
          (ex. native-country).
        - dropna=False garde les manquants comme modalité "(Manquant)", dont le
          taux peut lui-même être informatif (ex. occupation manquante -> surtout
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
        """Par variable numérique : nombre, taux et bornes des outliers (règle IQR).

        Une valeur est un outlier si elle est sous Q1 - whisker*IQR ou au-dessus
        de Q3 + whisker*IQR (whisker=1.5 = règle standard du boxplot). Trié par
        taux d'outliers décroissant.
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
        """Résumé console : dimensions, équilibre des classes, manquants, stats."""
        print(f"Dimensions du dataset : {self.shape()}")
        print(f"\nÉquilibre des classes :\n{self.class_balance()}\n")
        mv = self.missing_values()
        if mv.empty:
            print("Aucune valeur manquante détectée.")
        else:
            print(f"Valeurs manquantes :\n{mv}\n")
        print(f"\nStatistiques numériques :\n{self.numerical_stats()}")

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------

    def plot_class_balance(self, figsize: tuple = (12, 4)) -> plt.Figure:
        """Équilibre de la cible : barres (effectifs) + camembert (pourcentages)."""
        cb = self.class_balance()
        colors = ["steelblue", "tomato"]
        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=figsize)

        # Gauche : effectifs bruts
        ax_bar.bar(cb.index, cb["count"], color=colors)
        ax_bar.set_xlabel(TARGET)
        ax_bar.set_ylabel("effectif")
        for i, count in enumerate(cb["count"]):
            ax_bar.text(i, count, f"{count:,}", ha="center", va="bottom")

        # Droite : répartition en pourcentages
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
        """Vue bivariée des variable(s) numérique(s) vs la classe de revenu : une
        ligne par variable avec la distribution (gauche) et un boxplot par classe
        (droite). Les variables discrètes ont des bacs alignés sur les entiers,
        sans KDE.

        variable=None trace toutes les numériques ; passez un nom pour n'en
        tracer qu'une. filter_positive=True ne garde que les valeurs strictement
        positives (utile pour les capital zero-inflated), appliqué par variable.
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

            # Gauche : distribution ventilée par classe de revenu
            sns.histplot(data=sub, x=col, hue=TARGET, bins=bins,
                         kde=not is_discrete, palette=palette, ax=axes[i, 0])
            axes[i, 0].set_title(f"Distribution de {col} selon le revenu{suffix}",
                                 fontweight="bold")
            axes[i, 0].set_ylabel("Nombre de personnes")

            # Droite : boxplot par classe de revenu
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
        """Diagnostics univariés d'une variable numérique, côte à côte :
        distribution (hist + KDE), boxplot et QQ-plot normal.

        filter_positive=True ne garde que les valeurs strictement positives,
        utile pour les variables zero-inflated (capital-gain, capital-loss) dont
        le pic à zéro masque sinon la forme des valeurs réelles.
        """
        title = title or variable
        data = self.df[variable].dropna()
        if filter_positive:
            data = data[data > 0]

        # Les variables entières discrètes (ex. education-num) ont besoin d'un bac
        # par valeur, sinon des bacs fixes laissent des trous entre entiers (effet
        # "peigne"). Le KDE suppose aussi la continuité : on le retire dans ce cas.
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

        # 3. QQ-plot (vérification de la normalité)
        stats.probplot(data, dist="norm", plot=axes[2])
        axes[2].set_title(f"QQ-plot - {title}", fontweight="bold")

        suffix = " (valeurs > 0)" if filter_positive else ""
        fig.suptitle(f"Analyse univariée de {title}{suffix}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout()
        return fig

    def zero_proportion(self, variables: list[str] | None = None) -> pd.DataFrame:
        """Univarié : % global de valeurs nulles vs non nulles par variable.

        Pour les variables zero-inflated (capital-gain, capital-loss), quantifie
        la domination du pic à zéro, sans aucune ventilation par la cible.
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
        """Univarié : barre horizontale empilée part nulle vs non nulle par
        variable (sans ventilation par la cible)."""
        prop = self.zero_proportion(variables)
        fig, ax = plt.subplots(figsize=figsize)
        prop.plot(kind="barh", stacked=True, ax=ax, color=["lightgrey", "tomato"])
        ax.set_xlabel("% des lignes")
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
        """Bivarié : au sein de chaque classe de revenu, répartition nul vs non
        nul pour chaque variable (MultiIndex variable x classe de revenu).

        normalize=True renvoie des pourcentages par ligne (% dans la classe) ;
        normalize=False renvoie les effectifs bruts. Pensé pour les variables
        zero-inflated (capital-gain, capital-loss) : l'écart entre classes sur la
        colonne "> 0" est là où vit le signal discriminant.
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
        """Barres empilées part nulle vs non nulle par classe de revenu, un
        panneau par variable (par défaut les variables capital zero-inflated)."""
        variables = variables or ["capital-gain", "capital-loss"]
        table = self.zero_proportion_by_class(variables)
        fig, axes = plt.subplots(1, len(variables), figsize=figsize, squeeze=False)
        for ax, var in zip(axes.flat, variables):
            ct = table.loc[var]
            ct.plot(kind="bar", stacked=True, ax=ax,
                    color=["lightgrey", "tomato"], rot=0)
            ax.set_title(var)
            ax.set_ylabel("% dans la classe")
            ax.set_xlabel(TARGET)
            ax.legend(title=var, fontsize=8, loc="lower right")
            # Annote la part non nulle (la partie discriminante)
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
        """Distribution univariée de chaque variable catégorielle.

        Barres horizontales triées (idéal pour labels longs et modalités
        dominantes). Pour les variables à forte cardinalité (native-country,
        occupation, education), seules les top_n modalités sont affichées, le
        reste étant agrégé dans une barre "Autres (k)". Les manquants ont leur
        propre barre "(Manquant)". Les barres sont annotées du % de lignes.
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
            vc = vc.iloc[::-1]  # la plus grande en haut pour les barres horizontales
            ax.barh(vc.index.astype(str), vc.values, color="steelblue")
            for j, count in enumerate(vc.values):
                ax.text(count, j, f" {count / total * 100:.1f}%",
                        va="center", fontsize=8)
            ax.set_title(f"{col} ({self.X[col].nunique()} modalités)")
            ax.set_xlabel("effectif")
            ax.margins(x=0.12)  # place pour les étiquettes %
        for ax in axes.flat[len(CATEGORICAL_FEATURES):]:
            ax.set_visible(False)
        fig.suptitle("Distribution des variables catégorielles", y=1.005,
                     fontsize=14)
        fig.tight_layout()
        return fig

    def plot_categorical_bivariate(self, figsize: tuple = (20, 18)) -> plt.Figure:
        """Taux de >50K par modalité de chaque catégorielle, avec la ligne moyenne."""
        n_cols = 2
        n_rows = int(np.ceil(len(CATEGORICAL_FEATURES) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Taux global de >50K : une modalité au-dessus de cette ligne sur-représente
        # les hauts revenus, en-dessous elle les sous-représente. Loin de la ligne
        # = signal fort.
        baseline = (self.y == POSITIVE_CLASS).mean() * 100
        for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
            ct = pd.crosstab(self.X[col], self.y, normalize="index") * 100
            ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"],
                    rot=30, fontsize=8)
            ax.axhline(baseline, ls="--", color="tomato", lw=1.2,
                       label=f"moy. {POSITIVE_CLASS} ({baseline:.1f}%)")
            ax.set_title(col)
            ax.set_ylabel("% dans la catégorie")
            ax.legend(fontsize=7, loc="upper right")
        for ax in axes.flat[len(CATEGORICAL_FEATURES):]:
            ax.set_visible(False)
        fig.suptitle("Taux de revenu par catégorie", y=1.02)
        fig.tight_layout()
        return fig

    def plot_correlation_matrix(self, figsize: tuple = (10, 8)) -> plt.Figure:
        """Heatmap des corrélations de Pearson entre variables numériques."""
        corr = self.X[NUMERICAL_FEATURES].corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5,
            mask=np.triu(np.ones_like(corr, dtype=bool)),
        )
        ax.set_title("Corrélations entre variables numériques")
        fig.tight_layout()
        return fig

    def plot_cramers_v_matrix(
        self, include_target: bool = False, figsize: tuple = (9, 7)
    ) -> plt.Figure:
        """Heatmap de la matrice de Cramér's V entre variables catégorielles
        (en incluant éventuellement la colonne cible)."""
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
        """Heatmap du tableau de contingence entre deux variables catégorielles.

        normalize="index" -> P(col_b | col_a) conditionnelle par ligne en %, la
        façon la plus claire de lire quelle modalité de col_a correspond à quelle
        modalité de col_b (ex. comment marital-status se projette sur
        relationship). normalize=None -> effectifs bruts. Complète cramers_v : le
        V dit la force, ceci montre où.
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
        """Superpose le taux de >50K calculé par cat_col (catégoriel) et par
        num_col (numérique), pour prouver que les deux encodages définissent des
        groupes identiques.

        Les courbes coïncident car chaque modalité de cat_col correspond à
        exactement une valeur de num_col. Avec verbose=True, affiche aussi les
        deux vecteurs de taux et s'ils sont égaux.
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
        ax.set_title(f"{cat_col} vs {num_col} : taux >50K superposés "
                     f"(identiques : {identical})")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    def plot_missing_values(self, figsize: tuple = (8, 4)) -> plt.Figure:
        """Barres horizontales du pourcentage de manquants par variable."""
        mv = self.missing_values()
        if mv.empty:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, "Aucune valeur manquante", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(mv.index, mv["pct"])
        ax.set_xlabel("Manquants (%)")
        ax.set_title("Valeurs manquantes par variable")
        fig.tight_layout()
        return fig
