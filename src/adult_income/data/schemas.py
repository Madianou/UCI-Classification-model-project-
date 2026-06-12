NUMERICAL_FEATURES: list[str] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET: str = "income"
# Positive class after stripping whitespace/trailing dots
POSITIVE_CLASS: str = ">50K"
NEGATIVE_CLASS: str = "<=50K"
