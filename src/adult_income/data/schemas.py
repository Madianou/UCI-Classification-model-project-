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

# Asymetric features with many outliers that may benefit from RobustScaler instead of StandardScaler
ASIMMETRIC_NUMERICAL_FEATURES: list[str] = ["capital-gain", "capital-loss"]

# high cardinality categorical features that may benefit from target encoding instead of one-hot
HIGH_CARDINALITY_CATEGORICAL_FEATURES: list[str] = ["occupation", "native-country"]

# Dropped features (excluded from modelling, kept in the base lists above for EDA):
# - "education" is redundant with "education-num"
# - "fnlwgt" is a sampling weight that doesn't add predictive power
FEATURES_TO_DROP: list[str] = ["education", "fnlwgt"]

# ----------------------------------------------------------------------
# Derived feature groupings consumed by the preprocessing pipeline.
# FEATURES_TO_DROP is applied here, so the base lists above stay complete
# for exploratory data analysis while the pipeline only sees the kept ones.
# ----------------------------------------------------------------------

# Numerical features scaled with StandardScaler (excludes asymmetric ones and dropped ones)
NORMAL_NUMERICAL_FEATURES: list[str] = [
    f for f in NUMERICAL_FEATURES
    if f not in ASIMMETRIC_NUMERICAL_FEATURES and f not in FEATURES_TO_DROP
]

# Asymmetric numerical features still active after dropping
ASYM_NUMERICAL_ACTIVE: list[str] = [
    f for f in ASIMMETRIC_NUMERICAL_FEATURES if f not in FEATURES_TO_DROP
]

# Categorical features encoded with OneHot (excludes high-cardinality and dropped ones)
ONEHOT_CATEGORICAL_FEATURES: list[str] = [
    f for f in CATEGORICAL_FEATURES
    if f not in HIGH_CARDINALITY_CATEGORICAL_FEATURES and f not in FEATURES_TO_DROP
]

# High-cardinality categorical features still active after dropping (target-encoded)
HIGH_CARD_ACTIVE: list[str] = [
    f for f in HIGH_CARDINALITY_CATEGORICAL_FEATURES if f not in FEATURES_TO_DROP
]
