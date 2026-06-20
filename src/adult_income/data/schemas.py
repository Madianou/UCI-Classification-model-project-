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

# Human-readable descriptions for each column (keys match the hyphenated
# dataset column names, not the underscore variants used elsewhere).
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Âge de l'individu",
    "workclass": "Type d'employeur ou statut professionnel",
    "fnlwgt": "Poids statistique attribué à l'individu dans l'enquête (représentativité de la population)",
    "education": "Niveau d'éducation sous forme catégorielle",
    "education-num": "Niveau d'éducation sous forme numérique",
    "marital-status": "Statut marital de l'individu",
    "occupation": "Profession exercée",
    "relationship": "Lien familial ou statut dans le ménage",
    "race": "Catégorie ethnique déclarée",
    "sex": "Sexe de l'individu",
    "capital-gain": "Gains en capital déclarés sur l'année (investissement par ex)",
    "capital-loss": "Pertes en capital déclarées sur l'année (investissement par ex)",
    "hours-per-week": "Nombre d'heures travaillées par semaine",
    "native-country": "Pays d'origine",
    "income": "Classe de revenu annuel : <=50K ou >50K",
}

TARGET: str = "income"
# Positive class after stripping whitespace/trailing dots
POSITIVE_CLASS: str = ">50K"
NEGATIVE_CLASS: str = "<=50K"

# Asymetric features with many outliers that may benefit from RobustScaler instead of StandardScaler
ASIMMETRIC_NUMERICAL_FEATURES: list[str] = ["capital-gain", "capital-loss"]

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

# ----------------------------------------------------------------------
# EDA-aligned feature groupings (bilan §9 of the EDA notebook).
# These refine the generic groupings above with per-feature decisions,
# so the pipeline applies a tailored treatment instead of one-size-fits-all.
# ----------------------------------------------------------------------

# capital-gain : kept via log1p + RobustScaler (the heavy tail is predictive)
LOG_NUMERICAL_FEATURES: list[str] = ["capital-gain"]

# capital-gain & capital-loss : binary has_* flag (zero-inflated "hurdle" part).
# capital-loss gets ONLY this flag (its magnitude carries little extra signal);
# capital-gain gets both the flag and the log1p column above.
BINARY_NUMERICAL_FEATURES: list[str] = ["capital-gain", "capital-loss"]

# Low-cardinality categoricals with no informative missingness -> mode + encode
MODE_CATEGORICAL_FEATURES: list[str] = [
    "marital-status",
    "relationship",
    "race",
    "sex",
]

# Categoricals whose missingness is INFORMATIVE (~9.5% >50K vs 24% baseline):
# keep the signal with a dedicated "Unknown" category instead of imputing.
UNKNOWN_ONEHOT_FEATURES: list[str] = ["workclass"]    # low card -> OneHot
UNKNOWN_TARGET_FEATURES: list[str] = ["occupation"]   # high card -> TargetEncoder

# High-cardinality, NON-informative missingness -> impute mode + group rare modalities
GROUPED_CATEGORICAL_FEATURES: list[str] = ["native-country"]

# Modalities with fewer than this many observations are grouped together.
# Applied ONLY to native-country (~20 rare modalities), per the EDA bilan.
# Every other categorical keeps all its modalities, even rare ones.
RARE_CATEGORY_MIN_FREQUENCY: int = 50
