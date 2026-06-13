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
    ASYM_NUMERICAL_ACTIVE,
    ONEHOT_CATEGORICAL_FEATURES,
    HIGH_CARD_ACTIVE,
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
        """Return a fitted-ready ColumnTransformer."""

        normal_num = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                *([("scaler", StandardScaler())] if self.scaling else []),
            ]
        )
        asym_num = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "log1p",
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                ),
                *([("scaler", RobustScaler())] if self.scaling else []),
            ]
        )

        if self.encoding == "onehot":
            cat_encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                sparse_output=False,
            )
        else:
            cat_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )

        cat = Pipeline(
            [
                (
                    "replace_qm",
                    FunctionTransformer(
                        self.replace_questions_marks, feature_names_out="one-to-one"
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", cat_encoder),
            ]
        )

        high_card_cat = Pipeline(
            [
                (
                    "replace_qm",
                    FunctionTransformer(
                        self.replace_questions_marks, feature_names_out="one-to-one"
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TargetEncoder()),
            ]
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", normal_num, NORMAL_NUMERICAL_FEATURES),
                ("num_asym", asym_num, ASYM_NUMERICAL_ACTIVE),
                ("cat", cat, ONEHOT_CATEGORICAL_FEATURES),
                ("cat_high_card", high_card_cat, HIGH_CARD_ACTIVE),
            ],
            remainder="drop",
        )
        return self._preprocessor

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
