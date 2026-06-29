import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)

from ..data.schemas import (
    NORMAL_NUMERICAL_FEATURES,
    LOG_NUMERICAL_FEATURES,
    BINARY_NUMERICAL_FEATURES,
    MODE_CATEGORICAL_FEATURES,
    UNKNOWN_ONEHOT_FEATURES,
    GROUPED_CATEGORICAL_FEATURES,
    RARE_CATEGORY_MIN_FREQUENCY,
)


class PreprocessingPipeline:
    """
    Construit un ColumnTransformer sklearn pour le dataset Adult.

    Toutes les catégorielles sont encodées en one-hot ; cet encodage unique
    convient à tous les modèles du projet.

    Paramètres
    ----------
    scaling : bool
        Applique StandardScaler aux numériques normales (nécessaire pour LR,
        SVM, KNN) et RobustScaler au capital-gain log1p'é. Sans effet pour les
        modèles à arbres, invariants à l'échelle.

    Les trois flags suivants gardent les choix du bilan AED par défaut ; ils
    servent aux **ablations** du prétraitement (un facteur à la fois) :

    drop_fnlwgt : bool
        True (défaut) supprime fnlwgt ; False la garde comme numérique scalée.
    unknown_category : bool
        True (défaut) garde une catégorie dédiée "Unknown" pour la manquance
        informative de workclass/occupation ; False les impute par le mode.
    capital_treatment : "hurdle" | "naive"
        "hurdle" (défaut) = log1p(capital-gain) + indicateurs binaires has_* ;
        "naive" = capital-gain/loss en simples numériques scalées (ni log, ni flag).
    """

    def __init__(
        self,
        scaling: bool = True,
        drop_fnlwgt: bool = True,
        unknown_category: bool = True,
        capital_treatment: str = "hurdle",
    ) -> None:
        if capital_treatment not in {"hurdle", "naive"}:
            raise ValueError("capital_treatment doit valoir 'hurdle' ou 'naive'")
        self.scaling = scaling
        self.drop_fnlwgt = drop_fnlwgt
        self.unknown_category = unknown_category
        self.capital_treatment = capital_treatment
        self._preprocessor: ColumnTransformer | None = None
        self._label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Preprocessor (variables)
    # ------------------------------------------------------------------

    def build(self) -> ColumnTransformer:
        """Renvoie un ColumnTransformer prêt à être ajusté.

        Les branches suivent le bilan AED par défaut ; les flags du constructeur
        (``drop_fnlwgt``, ``unknown_category``, ``capital_treatment``) basculent
        les choix discutables pour les études d'ablation.
        """

        # --- Numériques ------------------------------------------------
        # Numériques normales : imputation médiane + StandardScaler. fnlwgt
        # rejoint cette branche si conservée, de même que capital-* en mode "naive".
        normal_features = list(NORMAL_NUMERICAL_FEATURES)
        if not self.drop_fnlwgt:
            normal_features.append("fnlwgt")
        if self.capital_treatment == "naive":
            normal_features += list(BINARY_NUMERICAL_FEATURES)

        normal_num = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                *([("scaler", StandardScaler())] if self.scaling else []),
            ]
        )
        transformers = [("num", normal_num, normal_features)]

        # Décomposition "hurdle" de capital-* : log1p(capital-gain) + flags has_*.
        if self.capital_treatment == "hurdle":
            log_num = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "log1p",
                        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    ),
                    *([("scaler", RobustScaler())] if self.scaling else []),
                ]
            )
            binary_num = Pipeline(
                [
                    (
                        "binarize",
                        FunctionTransformer(
                            self.binarize_positive,
                            feature_names_out=self._has_feature_names,
                        ),
                    ),
                ]
            )
            transformers += [
                ("num_log", log_num, LOG_NUMERICAL_FEATURES),
                ("num_bin", binary_num, BINARY_NUMERICAL_FEATURES),
            ]

        # --- Catégorielles ---------------------------------------------
        # Faible cardinalité, manquance non informative : imputation mode + encodage.
        cat_mode = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._make_cat_encoder()),
            ]
        )
        # workclass & occupation : manquance informative -> catégorie "Unknown"
        # (ou mode dans la variante d'ablation via ``unknown_category=False``) + OneHot.
        cat_unknown_onehot = Pipeline(
            [
                ("imputer", self._missing_imputer()),
                ("encoder", self._make_cat_encoder()),
            ]
        )
        # native-country : manquance non informative -> imputation mode ; ~20
        # modalités rares regroupées (min_frequency = RARE_CATEGORY_MIN_FREQUENCY).
        cat_grouped = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._make_cat_encoder(RARE_CATEGORY_MIN_FREQUENCY)),
            ]
        )

        transformers += [
            ("cat", cat_mode, MODE_CATEGORICAL_FEATURES),
            ("cat_unknown", cat_unknown_onehot, UNKNOWN_ONEHOT_FEATURES),
            ("cat_country", cat_grouped, GROUPED_CATEGORICAL_FEATURES),
        ]

        self._preprocessor = ColumnTransformer(transformers, remainder="drop")
        self._preprocessor.set_output(transform="pandas")
        return self._preprocessor

    def _missing_imputer(self) -> SimpleImputer:
        """Imputeur des colonnes à manquance informative (workclass/occupation).

        Catégorie dédiée "Unknown" par défaut ; imputation mode dans l'ablation.
        """
        if self.unknown_category:
            return SimpleImputer(strategy="constant", fill_value="Unknown")
        return SimpleImputer(strategy="most_frequent")

    def _make_cat_encoder(self, min_frequency: int | None = None):
        """Encodeur one-hot pour les catégorielles.

        ``min_frequency`` regroupe les modalités rares ; laissé à ``None``,
        chaque modalité garde sa colonne (seule native-country active le
        regroupement, selon le bilan AED). ``infrequent_if_exist`` protège tout
        de même contre les catégories inédites au transform, même sans regroupement.
        """
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
        )

    @staticmethod
    def _has_feature_names(transformer, input_features) -> np.ndarray:
        """Nomme les colonnes d'indicateur binaire has_<variable>."""
        return np.asarray([f"has_{name}" for name in input_features], dtype=object)

    @staticmethod
    def binarize_positive(df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie 1.0 si la valeur est strictement positive, sinon 0.0.

        Utilisé pour les variables zero-inflated capital-* : capte le signal
        gain/perte non nul indépendamment de la magnitude.
        """
        return (df > 0).astype(float)

    def plot_capital_treatment(
        self, X_raw: pd.DataFrame, X_transformed: pd.DataFrame
    ) -> tuple:
        """Graphe de vérification de la décomposition "hurdle" de capital-*.

        Panneau gauche : capital-gain brut (axe y log, queue lourde zero-inflated).
        Panneau droit  : la sortie log1p + RobustScaler -> même ordre de magnitude
        avec une queue compressée. Renvoie aussi la part de capital-gain/loss non
        nul captée par les flags has_*.

        Requiert ``capital_treatment="hurdle"`` (la matrice transformée doit
        contenir les colonnes ``num_log__`` / ``num_bin__``). Renvoie ``(fig, proportions)``.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.capital_treatment != "hurdle":
            raise ValueError(
                "plot_capital_treatment requiert capital_treatment='hurdle'"
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(X_raw["capital-gain"], ax=axes[0], bins=40, color="steelblue")
        axes[0].set_title("capital-gain - brut")
        axes[0].set_yscale("log")
        sns.histplot(
            X_transformed["num_log__capital-gain"],
            ax=axes[1],
            bins=40,
            color="seagreen",
        )
        axes[1].set_title("capital-gain - log1p + RobustScaler")
        fig.suptitle("capital-gain : log1p conserve la magnitude", fontsize=14)
        fig.tight_layout()

        bin_cols = [f"num_bin__has_{c}" for c in BINARY_NUMERICAL_FEATURES]
        proportions = X_transformed[bin_cols].mean().rename("proportion à 1").round(4)
        return fig, proportions

    def build_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        """Compose le preprocessor et un classifieur en une Pipeline complète."""
        preprocessor = self.build()
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

    # ------------------------------------------------------------------
    # Encodage de la cible
    # ------------------------------------------------------------------

    def fit_transform_target(self, y: pd.Series) -> np.ndarray:
        """Ajuste le LabelEncoder sur y et renvoie le tableau encodé."""
        return self._label_encoder.fit_transform(y)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Encode y avec le LabelEncoder déjà ajusté."""
        return self._label_encoder.transform(y)

    @property
    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

    @property
    def classes_(self) -> np.ndarray:
        return self._label_encoder.classes_

    @property
    def preprocessor(self) -> ColumnTransformer | None:
        return self._preprocessor
