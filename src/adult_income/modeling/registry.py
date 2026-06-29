from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def _make_knn_pca() -> Pipeline:
    """KNN précédé d'une PCA(20) pour contrer le fléau de la dimension.

    La PCA est dans l'estimateur, donc réajustée à chaque pli de CV (pas de
    fuite). L'entrée est dense (le one-hot du preprocessor a sparse_output=False).
    Comparé au ``knn`` simple pour mesurer l'apport réel de l'étape PCA.
    """
    return Pipeline(
        [
            ("pca", PCA(n_components=20, random_state=42)),
            ("knn", KNeighborsClassifier(n_jobs=-1)),
        ]
    )


def _make_stacking() -> StackingClassifier:
    """Stacking de trois familles complémentaires + méta-modèle logistique."""
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
    # Référence triviale (plancher) : prédit toujours la classe majoritaire.
    "dummy": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "adaboost": lambda: AdaBoostClassifier(random_state=42),
    # SVM linéaire : minimisation de la perte hinge L2-régularisée. On utilise
    # LinearSVC (solveur liblinear) plutôt que SVC(kernel="linear") : il scale
    # quasi-linéairement avec n_samples là où libsvm est en O(n²)-O(n³), donc
    # ~100× plus rapide sur Adult (~32k lignes). dual="auto" choisit la bonne
    # formulation (primal quand n_samples > n_features) ; max_iter relevé pour
    # éviter les ConvergenceWarning. LinearSVC n'expose pas predict_proba, d'où
    # le CalibratedClassifierCV qui ajoute des probabilités calibrées.
    "svm": lambda: CalibratedClassifierCV(
        LinearSVC(dual="auto", max_iter=5000, random_state=42), ensemble=False
    ),
    "knn": lambda: KNeighborsClassifier(n_jobs=-1),  # KNN simple (sans ACP)
    "knn_pca": _make_knn_pca,  # PCA(20) + KNN (à comparer au knn simple)
    "xgboost": lambda: XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
    "stacking": _make_stacking,
}


class ModelRegistry:
    """Catalogue central de tous les classifieurs du projet."""

    @classmethod
    def available(cls) -> list[str]:
        """Noms de tous les modèles du catalogue."""
        return list(_CATALOGUE)

    @classmethod
    def get(cls, name: str) -> BaseEstimator:
        """Renvoie une instance neuve (non entraînée) du modèle nommé."""
        if name not in _CATALOGUE:
            raise ValueError(f"Modèle inconnu {name!r}. Disponibles : {cls.available()}")
        return _CATALOGUE[name]()

    @classmethod
    def get_all(cls) -> dict[str, BaseEstimator]:
        """Renvoie une instance neuve par modèle enregistré."""
        return {name: _CATALOGUE[name]() for name in _CATALOGUE}

    @classmethod
    def get_subset(cls, names: list[str]) -> dict[str, BaseEstimator]:
        """Renvoie une instance neuve pour chaque nom de modèle demandé."""
        return {name: cls.get(name) for name in names}
