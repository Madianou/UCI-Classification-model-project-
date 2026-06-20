import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
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
    UNKNOWN_TARGET_FEATURES,
    GROUPED_CATEGORICAL_FEATURES,
    RARE_CATEGORY_MIN_FREQUENCY,
)


class PreprocessingPipeline:
    """
    Builds a sklearn ColumnTransformer for the Adult dataset.

    Parameters
    ----------
    encoding : "onehot" | "ordinal"
        Categorical encoding strategy. OneHot works best for linear models;
        ordinal is more compact for tree-based ones.
    scaling : bool
        Apply StandardScaler to numerical features (needed for LR, SVM, KNN)
        except capital-gain and capital-loss.
        Apply RobustScaler instead of StandardScaler if outliers are present.
        (because RobustScaler is better if there are outliers)
    """

    def __init__(self, encoding: str = "onehot", scaling: bool = True) -> None:
        if encoding not in {"onehot", "ordinal"}:
            raise ValueError("encoding must be 'onehot' or 'ordinal'")
        self.encoding = encoding
        self.scaling = scaling
        self._preprocessor: ColumnTransformer | None = None
        self._label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Preprocessor (features)
    # ------------------------------------------------------------------

    def build(self) -> ColumnTransformer:
        """Return a fitted-ready ColumnTransformer.

        The branches follow the EDA bilan: each feature group gets the
        treatment its distribution calls for, rather than a generic one.
        """

        # --- Numerical -------------------------------------------------
        # Normal numerics: median impute + StandardScaler.
        normal_num = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                *([("scaler", StandardScaler())] if self.scaling else []),
            ]
        )
        # capital-gain: log1p compresses the heavy tail, RobustScaler tames
        # the remaining outliers while preserving the (predictive) magnitude.
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
        # capital-gain & capital-loss: zero/non-zero "hurdle" flag has_*.
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

        # --- Categorical ----------------------------------------------
        # Low-cardinality, no informative missingness: mode impute + encode.
        cat_mode = Pipeline(
            [
                (
                    "replace_qm",
                    FunctionTransformer(
                        self.replace_questions_marks, feature_names_out="one-to-one"
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._make_cat_encoder()),
            ]
        )
        # workclass: missingness is informative -> dedicated "Unknown" category.
        cat_unknown_onehot = Pipeline(
            [
                (
                    "replace_qm", # au cas ou le nettoyage au load n'a pas suffit
                    FunctionTransformer(
                        self.replace_questions_marks, feature_names_out="one-to-one"
                    ),
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                ),
                ("encoder", self._make_cat_encoder()),
            ]
        )
        # occupation: informative missingness too, but high cardinality ->
        # "Unknown" category + TargetEncoder (cross-fitted, no leakage).
        cat_unknown_target = Pipeline(
            [
                (
                    "replace_qm", # au cas ou le nettoyage au load n'a pas suffit
                    FunctionTransformer(
                        self.replace_questions_marks, feature_names_out="one-to-one"
                    ),
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                ),
                ("encoder", TargetEncoder()),
            ]
        )
        # native-country: non-informative missingness -> mode impute; ~20 rare
        # modalities grouped together (min_frequency = RARE_CATEGORY_MIN_FREQUENCY).
        cat_grouped = Pipeline(
            [
                (
                    "replace_qm", # au cas ou le nettoyage au load n'a pas suffit
                    FunctionTransformer(
                        self.replace_questions_marks, feature_names_out="one-to-one"
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    self._make_cat_encoder(RARE_CATEGORY_MIN_FREQUENCY),
                ),
            ]
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", normal_num, NORMAL_NUMERICAL_FEATURES),
                ("num_log", log_num, LOG_NUMERICAL_FEATURES),
                ("num_bin", binary_num, BINARY_NUMERICAL_FEATURES),
                ("cat", cat_mode, MODE_CATEGORICAL_FEATURES),
                ("cat_unknown", cat_unknown_onehot, UNKNOWN_ONEHOT_FEATURES),
                ("cat_occupation", cat_unknown_target, UNKNOWN_TARGET_FEATURES),
                ("cat_country", cat_grouped, GROUPED_CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        return self._preprocessor

    def _make_cat_encoder(self, min_frequency: int | None = None):
        """OneHot (linear models) or Ordinal (trees) encoder.

        ``min_frequency`` groups rare modalities together; left at ``None``
        every modality keeps its own column (only native-country opts in to
        grouping, per the EDA bilan). ``infrequent_if_exist`` still protects
        against unseen categories at transform time even without grouping.
        """
        if self.encoding == "onehot":
            return OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_frequency,
                sparse_output=False,
            )
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            min_frequency=min_frequency,
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

    def replace_questions_marks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace '?' with np.nan across the columns of the given subset.

        Each ColumnTransformer branch only receives its own columns, so we
        operate on whatever is present instead of the global feature list.
        """
        return df.replace("?", np.nan)

    @property
    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

    @property
    def classes_(self) -> np.ndarray:
        return self._label_encoder.classes_

    @property
    def preprocessor(self) -> ColumnTransformer | None:
        return self._preprocessor
