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

# Ordre exact des colonnes dans les fichiers UCI bruts (adult.data /
# adult.test), qui sont sans en-tête. Diffère de ALL_FEATURES (qui groupe les
# numériques en premier) ; DataLoader s'en sert pour nommer les colonnes du CSV.
RAW_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Descriptions lisibles de chaque colonne (les clés reprennent les noms du
# dataset avec tirets, pas les variantes avec underscore utilisées ailleurs).
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
# Classe positive après retrait des espaces / points finaux
POSITIVE_CLASS: str = ">50K"
NEGATIVE_CLASS: str = "<=50K"

# Variables asymétriques à nombreux outliers : RobustScaler plutôt que StandardScaler
ASYMMETRIC_NUMERICAL_FEATURES: list[str] = ["capital-gain", "capital-loss"]

# Variables supprimées (exclues de la modélisation, gardées dans les listes
# ci-dessus pour l'AED) :
# - "education" est redondante avec "education-num"
# - "fnlwgt" est un poids d'échantillonnage sans pouvoir prédictif
FEATURES_TO_DROP: list[str] = ["education", "fnlwgt"]

# ----------------------------------------------------------------------
# Regroupements de variables dérivés, consommés par la pipeline de prétraitement.
# FEATURES_TO_DROP est appliqué ici, donc les listes de base restent complètes
# pour l'AED tandis que la pipeline ne voit que les variables conservées.
# ----------------------------------------------------------------------

# Numériques mises à l'échelle par StandardScaler (hors asymétriques et droppées)
NORMAL_NUMERICAL_FEATURES: list[str] = [
    f for f in NUMERICAL_FEATURES
    if f not in ASYMMETRIC_NUMERICAL_FEATURES and f not in FEATURES_TO_DROP
]

# ----------------------------------------------------------------------
# Regroupements alignés sur l'AED (bilan §9 du notebook d'AED).
# Ils affinent les regroupements génériques ci-dessus par des décisions
# par variable, pour un traitement sur mesure plutôt qu'uniforme.
# ----------------------------------------------------------------------

# capital-gain : conservée via log1p + RobustScaler (la queue lourde est prédictive)
LOG_NUMERICAL_FEATURES: list[str] = ["capital-gain"]

# capital-gain & capital-loss : indicateur binaire has_* (partie « hurdle » des
# variables zero-inflated). capital-loss n'a QUE cet indicateur (sa magnitude
# apporte peu) ; capital-gain a l'indicateur ET la colonne log1p ci-dessus.
BINARY_NUMERICAL_FEATURES: list[str] = ["capital-gain", "capital-loss"]

# Catégorielles à faible cardinalité, manquance non informative -> mode + encodage
MODE_CATEGORICAL_FEATURES: list[str] = [
    "marital-status",
    "relationship",
    "race",
    "sex",
]

# Catégorielles dont la manquance est INFORMATIVE (~9,5 % de >50K vs 24 % de base) :
# on garde le signal via une catégorie dédiée "Unknown" au lieu d'imputer.
# Les deux sont encodées en one-hot (les 15 modalités d'occupation restent gérables).
UNKNOWN_ONEHOT_FEATURES: list[str] = ["workclass", "occupation"]

# Forte cardinalité, manquance NON informative -> imputation mode + regroupement des rares
GROUPED_CATEGORICAL_FEATURES: list[str] = ["native-country"]

# Les modalités sous ce nombre d'observations sont regroupées ensemble.
# Appliqué UNIQUEMENT à native-country (~20 modalités rares), selon le bilan AED.
# Toutes les autres catégorielles gardent leurs modalités, même rares.
RARE_CATEGORY_MIN_FREQUENCY: int = 50
