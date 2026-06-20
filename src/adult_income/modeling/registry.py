from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def _make_knn() -> Pipeline:
    """KNN preceded by a PCA(20) to fight the curse of dimensionality.

    The PCA lives inside the estimator so it is refit on each CV fold (no
    leakage). Input is dense (the preprocessor's OneHot uses sparse_output=False).
    """
    return Pipeline(
        [
            ("pca", PCA(n_components=20, random_state=42)),
            ("knn", KNeighborsClassifier(n_jobs=-1)),
        ]
    )


def _make_stacking() -> StackingClassifier:
    """Stacking of three complementary families with a logistic meta-model."""
    return StackingClassifier(
        estimators=[
            ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
            ("tree", DecisionTreeClassifier(random_state=42)),
            ("boosting", GradientBoostingClassifier(random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1,
    )


_CATALOGUE: dict[str, type] = {
    # Trivial baseline (floor): always predicts the majority class.
    "dummy": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "adaboost": lambda: AdaBoostClassifier(random_state=42),
    "svm": lambda: SVC(probability=True, random_state=42),
    "knn": _make_knn,  # PCA(20) + KNN
    "naive_bayes": lambda: GaussianNB(),
    "xgboost": lambda: XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
    "lightgbm": lambda: LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    # allow_writing_files=False : sinon CatBoost tente de créer un dossier
    # catboost_info/ pour ses logs, ce qui échoue en CV parallèle (collisions).
    "catboost": lambda: CatBoostClassifier(
        random_state=42, verbose=0, allow_writing_files=False
    ),
    "stacking": _make_stacking,
}


class ModelRegistry:
    """Central catalogue of all classifiers used in the project."""

    @classmethod
    def available(cls) -> list[str]:
        return list(_CATALOGUE)

    @classmethod
    def get(cls, name: str) -> BaseEstimator:
        """Return a fresh (unfitted) estimator by name."""
        if name not in _CATALOGUE:
            raise ValueError(
                f"Unknown model {name!r}. Available: {cls.available()}"
            )
        return _CATALOGUE[name]()

    @classmethod
    def get_all(cls) -> dict[str, BaseEstimator]:
        """Return one fresh instance per registered model."""
        return {name: _CATALOGUE[name]() for name in _CATALOGUE}

    @classmethod
    def get_subset(cls, names: list[str]) -> dict[str, BaseEstimator]:
        return {name: cls.get(name) for name in names}
