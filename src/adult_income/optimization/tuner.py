import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Search space per model — each callable takes a Trial and returns a param dict
SearchSpace = Callable[[optuna.Trial], dict[str, Any]]

SEARCH_SPACES: dict[str, SearchSpace] = {
    "logistic_regression": lambda t: {
        "C": t.suggest_float("C", 1e-3, 100.0, log=True),
        "solver": t.suggest_categorical("solver", ["lbfgs", "saga"]),
    },
    "decision_tree": lambda t: {
        "max_depth": t.suggest_int("max_depth", 2, 30),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "criterion": t.suggest_categorical("criterion", ["gini", "entropy"]),
    },
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 300),
        "max_depth": t.suggest_int("max_depth", 3, 30),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 15),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2"]),
    },
    "gradient_boosting": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 250),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
    },
    "adaboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 400),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 2.0, log=True),
    },
    "knn": lambda t: {
        "knn__n_neighbors": t.suggest_int("knn__n_neighbors", 3, 50),
        "knn__weights": t.suggest_categorical("knn__weights", ["uniform", "distance"]),
        "knn__metric": t.suggest_categorical("knn__metric", ["euclidean", "manhattan"]),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 300),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
    },
    "svm": lambda t: {
        "estimator__C": t.suggest_float("estimator__C", 1e-2, 100.0, log=True),
    },
}


class HyperparameterTuner:
    """
    Optuna-based hyperparameter optimisation.

    Usage
    -----
    tuner = HyperparameterTuner(n_trials=50, scoring="roc_auc")
    best_params = tuner.tune("xgboost", pipeline, X_train, y_train)
    fig = tuner.plot_optimization_history("xgboost")

    Persistance
    -----------
    En passant ``storage=<dossier>``, chaque étude est écrite dans une base
    SQLite (toutes les trials, valeurs intermédiaires, pruning) et les
    meilleurs paramètres sont sauvegardés en JSON. On peut alors reprendre une
    optimisation interrompue, ou recharger les résultats dans une autre session
    sans tout relancer :

    tuner = HyperparameterTuner(n_trials=50, storage="artifacts/optuna")
    tuner.tune("xgboost", pipeline, X_train, y_train)   # écrit sur disque
    ...
    # plus tard, dans un autre notebook :
    tuner = HyperparameterTuner(storage="artifacts/optuna")
    study = tuner.load_study("xgboost")                 # relit depuis le disque
    best = HyperparameterTuner.load_best_params("artifacts/optuna")
    """

    DB_NAME = "optuna_studies.db"
    JSON_NAME = "best_params.json"

    def __init__(
        self,
        n_trials: int = 50,
        cv: int = 5,
        scoring: str = "roc_auc",
        n_jobs: int = 1,
        timeout: float | None = None,
        pruning: bool = True,
        storage: str | Path | None = None,
    ) -> None:
        # n_jobs : nombre de trials Optuna exécutés en parallèle.
        #   - n_jobs=1  -> trials séquentiels, le modèle garde son n_jobs=-1.
        #   - n_jobs>1  -> trials en parallèle ; le modèle est bridé à 1 thread
        #     (cf. _single_thread_overrides) pour éviter la sur-souscription.
        #   - n_jobs=-1 -> autant de trials parallèles que de cœurs.
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.pruning = pruning
        self.studies: dict[str, optuna.Study] = {}
        self.best_params: dict[str, dict] = {}

        # Persistance optionnelle : un dossier qui contiendra la base SQLite
        # (études Optuna) et le JSON des meilleurs paramètres.
        self.storage_dir = Path(storage) if storage is not None else None
        self._db_url: str | None = None
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            # URL SQLAlchemy attendue par Optuna ; chemin absolu pour éviter
            # toute ambiguïté selon le répertoire de travail du notebook.
            db_path = (self.storage_dir / self.DB_NAME).resolve()
            self._db_url = f"sqlite:///{db_path}"

    @staticmethod
    def tunable_models() -> list[str]:
        """Modèles disposant d'un espace de recherche (donc optimisables).

        Exclut les modèles du registry sans hyperparamètres à régler
        (``dummy``) ou volontairement non tunés ici (``stacking``).
        """
        return list(SEARCH_SPACES)

    def tune(
        self,
        model_name: str,
        pipeline,  # full sklearn Pipeline: preprocessor + classifier
        X,
        y,
    ) -> dict[str, Any]:
        """Run Optuna study and return best hyperparameters."""
        if model_name not in SEARCH_SPACES:
            raise ValueError(
                f"No search space defined for {model_name!r}. "
                f"Available: {list(SEARCH_SPACES)}"
            )

        scorer = get_scorer(self.scoring)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        y_arr = np.asarray(y)

        # Garde-fou contre la sur-souscription : si les trials Optuna tournent en
        # parallèle (n_jobs != 1), on force le modèle à un seul thread. Sinon
        # n_jobs trials × n_jobs threads modèle saturent le CPU et ralentissent.
        # Avec des trials séquentiels, le modèle garde son parallélisme (n_jobs=-1).
        single_thread = (
            self._single_thread_overrides(pipeline) if self.n_jobs != 1 else {}
        )

        def objective(trial: optuna.Trial) -> float:
            params = SEARCH_SPACES[model_name](trial)
            # Prefix params with 'classifier__' to target the pipeline step
            prefixed = {f"classifier__{k}": v for k, v in params.items()}
            prefixed.update(single_thread)

            # CV manuel fold-par-fold : on rapporte le score cumulé à Optuna
            # après chaque fold pour permettre au pruner d'abandonner tôt.
            fold_scores: list[float] = []
            for step, (tr_idx, val_idx) in enumerate(skf.split(X, y_arr)):
                X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
                X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
                candidate = clone(pipeline)
                candidate.set_params(**prefixed)
                candidate.fit(X_tr, y_arr[tr_idx])
                fold_scores.append(float(scorer(candidate, X_val, y_arr[val_idx])))

                trial.report(float(np.mean(fold_scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            study_name=self._study_name(model_name),
            storage=self._db_url,  # None => étude en mémoire (comme avant)
            load_if_exists=True,  # reprend l'étude déjà sauvegardée
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=(
                optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
                if self.pruning
                else optuna.pruners.NopPruner()
            ),
        )

        # Avec un stockage persistant, l'étude peut déjà contenir des trials :
        # on ne lance que celles qui manquent pour atteindre n_trials.
        remaining = max(self.n_trials - len(study.trials), 0)
        if remaining > 0:
            study.optimize(
                objective,
                n_trials=remaining,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True,
            )

        self.studies[model_name] = study
        self.best_params[model_name] = study.best_params
        if self.storage_dir is not None:
            self._save_best_params()
        return study.best_params

    def _study_name(self, model_name: str) -> str:
        """Nom d'étude qui inclut la métrique, pour ne pas mélanger des trials
        optimisés sur des scorings différents (ex. roc_auc vs average_precision)
        quand on réutilise le même storage."""
        return f"{model_name}__{self.scoring}"

    @staticmethod
    def _single_thread_overrides(pipeline) -> dict[str, Any]:
        """Params à forcer à 1 pour brider le parallélisme interne du modèle.

        Selon la lib, le paramètre de threads diffère (``n_jobs`` pour sklearn /
        XGBoost, ``thread_count`` pour CatBoost). On ne retient que les clés
        réellement exposées par ``get_params()`` du classifieur : le wrapper
        sklearn de XGBoost accepte n'importe quel kwarg via ``set_params`` sans
        lever, puis le repasse à ``fit`` qui l'ignore en émettant un warning
        ("Parameters: { thread_count } are not used"). Le test d'appartenance
        évite ce faux positif. Bride la sur-souscription quand les trials Optuna
        tournent en parallèle.
        """
        classifier = pipeline.named_steps.get("classifier")
        params = classifier.get_params()
        overrides: dict[str, Any] = {}
        for name in ("n_jobs", "thread_count"):
            if name not in params:
                continue
            # LogisticRegression.n_jobs n'a plus aucun effet (sklearn >= 1.8) et
            # émet un FutureWarning : inutile de le brider, on ne le passe pas.
            if name == "n_jobs" and isinstance(classifier, LogisticRegression):
                continue
            overrides[f"classifier__{name}"] = 1
        return overrides

    def plot_optimization_history(self, model_name: str):
        self._check_study(model_name)
        return optuna.visualization.plot_optimization_history(self.studies[model_name])

    def plot_param_importances(self, model_name: str):
        self._check_study(model_name)
        return optuna.visualization.plot_param_importances(self.studies[model_name])

    def plot_parallel_coordinate(self, model_name: str):
        self._check_study(model_name)
        return optuna.visualization.plot_parallel_coordinate(self.studies[model_name])

    def summary(self) -> None:
        for name, params in self.best_params.items():
            score = self.studies[name].best_value
            print(f"{name:25s} | best {self.scoring}: {score:.4f} | params: {params}")

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def _save_best_params(self) -> Path:
        """Écrit un JSON récapitulatif (meilleurs params + score) sur disque."""
        payload = {
            name: {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "scoring": self.scoring,
                "n_trials": len(study.trials),
            }
            for name, study in self.studies.items()
        }
        path = self.storage_dir / self.JSON_NAME
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load_study(self, model_name: str) -> optuna.Study:
        """Recharge une étude sauvegardée depuis la base SQLite."""
        if self._db_url is None:
            raise ValueError("No storage configured; pass storage=<dir> to load.")
        study = optuna.load_study(
            study_name=self._study_name(model_name), storage=self._db_url
        )
        self.studies[model_name] = study
        self.best_params[model_name] = study.best_params
        return study

    def list_saved_studies(self) -> list[str]:
        """Noms des études disponibles dans la base SQLite."""
        if self._db_url is None:
            return []
        return [s.study_name for s in optuna.get_all_study_summaries(self._db_url)]

    @classmethod
    def load_best_params(cls, storage: str | Path) -> dict[str, dict]:
        """Relit le JSON des meilleurs paramètres sans toucher à Optuna.

        Pratique pour réutiliser les hyperparamètres dans un autre notebook ou
        dans l'app Streamlit, sans rejouer l'optimisation.
        """
        path = Path(storage) / cls.JSON_NAME
        if not path.exists():
            raise FileNotFoundError(f"No saved params at {path}")
        return json.loads(path.read_text())

    def _check_study(self, model_name: str) -> None:
        if model_name not in self.studies:
            # Tente de recharger depuis le disque avant d'abandonner.
            if (
                self._db_url is not None
                and self._study_name(model_name) in self.list_saved_studies()
            ):
                self.load_study(model_name)
                return
            raise ValueError(f"No study found for {model_name!r}. Run tune() first.")
