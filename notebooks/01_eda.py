# 1 — Analyse exploratoire des données (EDA)

from adult_income import DataLoader, EDAAnalyzer

loader = DataLoader()
X, y = loader.load_train()  # AED sur le TRAIN uniquement ; le test reste de côté

print(f"Données chargées (train) : {X.shape[0]} observations, {X.shape[1]} features")
print(loader.class_distribution(y))

eda = EDAAnalyzer(X, y)
eda.summary()

eda.plot_class_balance()

eda.plot_missing_values()

eda.plot_numerical_bivariate()

eda.plot_categorical_bivariate()

eda.plot_correlation_matrix()
