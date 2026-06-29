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
    Builds a sklearn ColumnTransformer for the Adult dataset.

    All categorical features are one-hot encoded; this single encoding works
    for every model in the project.

    Parameters
    ----------
    scaling : bool
        Apply StandardScaler to the normal numerical features (needed for LR,
        SVM, KNN) and RobustScaler to the log1p'd capital-gain. Harmless for
        tree-based models, which are scale-invariant.

    The next three flags keep the EDA-bilan choices by default; they exist to
    run preprocessing **ablations** (one factor at a time):

    drop_fnlwgt : bool
        True (default) drops fnlwgt; False keeps it as a scaled numeric.
    unknown_category : bool
        True (default) keeps a dedicated "Unknown" category for the informative
        missingness of workclass/occupation; False imputes them by the mode.
    capital_treatment : "hurdle" | "naive"
        "hurdle" (default) = log1p(capital-gain) + has_* binary flags ;
        "naive" = capital-gain/loss as plain scaled numerics (no log, no flag).
    """

    def __init__(
        self,
        scaling: bool = True,
        drop_fnlwgt: bool = True,
        unknown_category: bool = True,
        capital_treatment: str = "hurdle",
    ) -> None:
        if capital_treatment not in {"hurdle", "naive"}:
            raise ValueError("capital_treatment must be 'hurdle' or 'naive'")
        self.scaling = scaling
        self.drop_fnlwgt = drop_fnlwgt
        self.unknown_category = unknown_category
        self.capital_treatment = capital_treatment
        self._preprocessor: ColumnTransformer | None = None
        self._label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Preprocessor (features)
    # ------------------------------------------------------------------

    def build(self) -> ColumnTransformer:
        """Return a fitted-ready ColumnTransformer.

        Branches follow the EDA bilan by default; the constructor flags
        (``drop_fnlwgt``, ``unknown_category``, ``capital_treatment``) toggle
        the debatable choices for ablation studies.
        """

        # --- Numerical -------------------------------------------------
        # Normal numerics: median impute + StandardScaler. fnlwgt joins this
        # branch when kept, and so do capital-* in the "naive" mode.
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

        # capital-* "hurdle" decomposition: log1p(capital-gain) + has_* flags.
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

        # --- Categorical ----------------------------------------------
        # Low-cardinality, no informative missingness: mode impute + encode.
        cat_mode = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._make_cat_encoder()),
            ]
        )
        # workclass & occupation: informative missingness -> "Unknown" category
        # (or mode in the ablation variant via ``unknown_category=False``) + OneHot.
        cat_unknown_onehot = Pipeline(
            [
                ("imputer", self._missing_imputer()),
                ("encoder", self._make_cat_encoder()),
            ]
        )
        # native-country: non-informative missingness -> mode impute; ~20 rare
        # modalities grouped together (min_frequency = RARE_CATEGORY_MIN_FREQUENCY).
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
        """Imputer for the informative-missingness columns (workclass/occupation).

        Dedicated "Unknown" category by default; mode imputation in the ablation.
        """
        if self.unknown_category:
            return SimpleImputer(strategy="constant", fill_value="Unknown")
        return SimpleImputer(strategy="most_frequent")

    def _make_cat_encoder(self, min_frequency: int | None = None):
        """One-hot encoder for categoricals.

        ``min_frequency`` groups rare modalities together; left at ``None``
        every modality keeps its own column (only native-country opts in to
        grouping, per the EDA bilan). ``infrequent_if_exist`` still protects
        against unseen categories at transform time even without grouping.
        """
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
        )

    @staticmethod
    def _has_feature_names(transformer, input_features) -> np.ndarray:
        """Name binary flag columns has_<feature>."""
        return np.asarray([f"has_{name}" for name in input_features], dtype=object)

    @staticmethod
    def binarize_positive(df: pd.DataFrame) -> pd.DataFrame:
        """Return 1.0 where the value is strictly positive, else 0.0.

        Used for the zero-inflated capital-* features: captures the
        has-any-gain/loss signal independently of the magnitude.
        """
        return (df > 0).astype(float)

    def plot_capital_treatment(
        self, X_raw: pd.DataFrame, X_transformed: pd.DataFrame
    ) -> tuple:
        """Verification plot of the capital-* "hurdle" decomposition.

        Left panel  : raw capital-gain (log y-scale, zero-inflated heavy tail).
        Right panel : the log1p + RobustScaler output -> same magnitude ordering
        with a compressed tail. Also returns the share of non-zero
        capital-gain/loss captured by the has_* flags.

        Requires ``capital_treatment="hurdle"`` (the transformed frame must hold
        the ``num_log__`` / ``num_bin__`` columns). Returns ``(fig, proportions)``.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.capital_treatment != "hurdle":
            raise ValueError(
                "plot_capital_treatment requires capital_treatment='hurdle'"
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
        """Compose the preprocessor with a classifier into a full Pipeline."""
        preprocessor = self.build()
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

    # ------------------------------------------------------------------
    # Target encoding
    # ------------------------------------------------------------------

    def fit_transform_target(self, y: pd.Series) -> np.ndarray:
        """Fit LabelEncoder on y and return encoded array."""
        return self._label_encoder.fit_transform(y)

    def transform_target(self, y: pd.Series) -> np.ndarray:
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
