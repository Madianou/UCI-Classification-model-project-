# Projet AI28 — Adult Income

Projet de machine learning sur le dataset [Adult Income (UCI)](https://archive.ics.uci.edu/dataset/2/adult).
Objectif : prédire si le revenu annuel d'un individu dépasse 50 K$ (classification binaire).

## Installation

```bash
uv sync --group dev   # installe toutes les dépendances + jupyter
```

```bash
uv run streamlit run app/streamlit_app.py # lance l'app streamlit
```

## Structure

```
src/adult_income/
├── data/
│   ├── schemas.py        # constantes : noms des features, TARGET, classes
│   └── loader.py         # DataLoader — lit les fichiers UCI locaux (split officiel / aléatoire)
├── eda/
│   └── analyzer.py       # EDAAnalyzer — stats descriptives, distributions, corrélations
├── preprocessing/
│   └── pipeline.py       # PreprocessingPipeline — ColumnTransformer + LabelEncoder cible
├── modeling/
│   └── registry.py       # ModelRegistry — catalogue de 10 classifieurs sklearn/xgb
├── optimization/
│   └── tuner.py          # HyperparameterTuner — optimisation Optuna (TPE), search spaces
└── evaluation/
    └── evaluator.py      # ModelEvaluator — CV, métriques test, ROC, heatmap comparaison

notebooks/
├── eda.ipynb            # Analyse exploratoire (sur le train uniquement)
├── preprocessing.ipynb  # Effets de la pipeline de prétraitement
├── modeling.ipynb       # Comparaison CV des modèles + ablations + finalistes
└── (évaluation finale sur le test : notebook dédié, après l'hyper-optimisation)
```

## Utilisation rapide

```python
from adult_income import (
    DataLoader, EDAAnalyzer, PreprocessingPipeline,
    ModelRegistry, HyperparameterTuner, ModelEvaluator,
)

# 1. Données
X_train, X_test, y_train, y_test = DataLoader().load_and_split()

# 2. Preprocessing
pp = PreprocessingPipeline(scaling=True)
y_train_enc = pp.fit_transform_target(y_train)
y_test_enc  = pp.transform_target(y_test)

# 3. Pipelines complets
pipelines = {name: pp.build_pipeline(est) for name, est in ModelRegistry.get_all().items()}

# 4. Évaluation CV
evaluator = ModelEvaluator(cv=5)
evaluator.cross_validate_all(pipelines, X_train, y_train_enc)
evaluator.cv_comparison_table()

# 5. Hyperopt (Optuna)
tuner = HyperparameterTuner(n_trials=50)
best = tuner.tune("xgboost", pipelines["xgboost"], X_train, y_train_enc)
```

## Modèles inclus

Dummy (baseline) · Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · AdaBoost · SVM · KNN (PCA + KNN) · XGBoost · Stacking
