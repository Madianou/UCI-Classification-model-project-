# 1 — Analyse exploratoire des données (EDA)

from adult_income import DataLoader, EDAAnalyzer

loader = DataLoader(test_size=0.2, random_state=42)
X, y = loader.load()

print(f"Données chargées : {X.shape[0]} observations, {X.shape[1]} features")
print(loader.class_distribution(y))

eda = EDAAnalyzer(X, y)
eda.summary()

eda.plot_class_balance()

eda.plot_missing_values()

eda.plot_numerical_distributions()

eda.plot_categorical_distributions()

eda.plot_correlation_matrix()
