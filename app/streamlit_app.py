"""
Démo interactive — Adult Income (UCI)

Lancement :
    uv run streamlit run app/streamlit_app.py

Deux onglets :
  1. Jeu de données  : exploration interactive (Plotly), inspirée de l'EDA.
  2. Modèle          : choix du modèle + hyperparamètres (manuels ou optimaux
                       Optuna), entraînement, métriques et graphiques.

Réutilise les modules du package : DataLoader, EDAAnalyzer, PreprocessingPipeline,
ModelRegistry, ModelEvaluator.
"""

import json
import sys
from pathlib import Path

# Permet d'importer le package src/ quand on lance streamlit depuis la racine
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from adult_income import (
    DataLoader,
    EDAAnalyzer,
    ModelEvaluator,
    ModelRegistry,
    PreprocessingPipeline,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    POSITIVE_CLASS,
    TARGET,
)

st.set_page_config(page_title="Adult Income — Démo", layout="wide")

# Fichier des hyperparamètres optimaux calculés par Optuna (scoring = AP).
BEST_PARAMS_FILE = ROOT / "artifacts" / "optuna_ap" / "best_params.json"


# ----------------------------------------------------------------------
# Chargements mis en cache (évitent de recharger/recalculer à chaque clic)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données…")
def load_data():
    loader = DataLoader()
    return loader.load_and_split()


@st.cache_data(show_spinner=False)
def load_best_params() -> dict:
    """Hyperparamètres optimaux Optuna ; {} si le fichier est absent."""
    if BEST_PARAMS_FILE.exists():
        return json.loads(BEST_PARAMS_FILE.read_text())
    return {}


X_train, X_test, y_train, y_test = load_data()
df_train = X_train.copy()
df_train[TARGET] = y_train.values

BEST_PARAMS = load_best_params()
analyzer = EDAAnalyzer(X_train, y_train)
baseline_rate = (y_train == POSITIVE_CLASS).mean() * 100


@st.cache_data(show_spinner="Calcul des associations catégorielles (Cramér's V)…")
def cramers_matrix():
    return analyzer.cramers_v_matrix(include_target=True)


# ----------------------------------------------------------------------
# Spécification des hyperparamètres exposés par modèle (mode manuel)
#   chaque entrée : (type, kwargs du widget)  -> construit dans la sidebar
#   Les clés sont les NOMS RÉELS passés à set_params (knn préfixé "knn__"
#   car le registre renvoie un Pipeline PCA + KNN).
# ----------------------------------------------------------------------
HYPERPARAMS = {
    "logistic_regression": {
        "C": ("slider_float", dict(min_value=0.01, max_value=10.0, value=1.0, step=0.01)),
        "penalty": ("select", dict(options=["l2", "l1"], index=0)),
    },
    "decision_tree": {
        "max_depth": ("slider_int", dict(min_value=1, max_value=30, value=10)),
        "min_samples_split": ("slider_int", dict(min_value=2, max_value=50, value=2)),
        "criterion": ("select", dict(options=["gini", "entropy"], index=0)),
    },
    "random_forest": {
        "n_estimators": ("slider_int", dict(min_value=10, max_value=500, value=100, step=10)),
        "max_depth": ("slider_int", dict(min_value=1, max_value=40, value=20)),
        "min_samples_split": ("slider_int", dict(min_value=2, max_value=50, value=2)),
    },
    "gradient_boosting": {
        "n_estimators": ("slider_int", dict(min_value=10, max_value=500, value=100, step=10)),
        "learning_rate": ("slider_float", dict(min_value=0.01, max_value=1.0, value=0.1, step=0.01)),
        "max_depth": ("slider_int", dict(min_value=1, max_value=10, value=3)),
    },
    "xgboost": {
        "n_estimators": ("slider_int", dict(min_value=10, max_value=500, value=100, step=10)),
        "learning_rate": ("slider_float", dict(min_value=0.01, max_value=1.0, value=0.3, step=0.01)),
        "max_depth": ("slider_int", dict(min_value=1, max_value=15, value=6)),
    },
    "knn": {
        "knn__n_neighbors": ("slider_int", dict(min_value=1, max_value=50, value=5)),
        "knn__weights": ("select", dict(options=["uniform", "distance"], index=0)),
    },
}


def build_widget(label, kind, kwargs):
    """Crée un widget sidebar selon le type déclaré."""
    if kind in ("slider_int", "slider_float"):
        return st.sidebar.slider(label, **kwargs)
    if kind == "select":
        return st.sidebar.selectbox(label, **kwargs)
    raise ValueError(kind)


# ----------------------------------------------------------------------
# Onglets
# ----------------------------------------------------------------------
tab_data, tab_model = st.tabs(["📊 Jeu de données", "🤖 Modèle & hyperparamètres"])


# ======================================================================
# Onglet 1 — Exploration du jeu de données (style EDA)
# ======================================================================
with tab_data:
    st.header("Jeu de données — Adult Income (UCI)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (train)", f"{len(df_train):,}")
    c2.metric("Variables", f"{X_train.shape[1]}")
    c3.metric("Taux >50K", f"{baseline_rate:.1f} %")
    c4.metric("Valeurs manquantes", f"{int(X_train.isnull().sum().sum()):,}")

    sub_overview, sub_num, sub_cat, sub_corr, sub_scatter = st.tabs(
        ["Vue d'ensemble", "Numériques", "Catégorielles", "Corrélations", "Nuage de points"]
    )

    # ------------------------------------------------------------------
    # Vue d'ensemble : aperçu, dictionnaire des variables, cible, manquants
    # ------------------------------------------------------------------
    with sub_overview:
        st.subheader("Aperçu")
        n = st.slider("Nombre de lignes affichées", 5, 100, 20, key="nrows")
        st.dataframe(df_train.head(n), width="stretch")

        st.subheader("Dictionnaire des variables")
        overview = analyzer.variable_overview()
        # "Nombre de modalités" mélange int (catégorielles) et "" (numériques) :
        # homogénéiser en texte, sinon Arrow refuse de sérialiser la colonne.
        overview["Nombre de modalités"] = overview["Nombre de modalités"].astype(str)
        st.dataframe(overview, width="stretch", hide_index=True)

        g1, g2 = st.columns(2)

        # Équilibre de la classe cible (barres + camembert)
        with g1:
            st.subheader("Distribution de la cible")
            cb = analyzer.class_balance()
            fig_cb = go.Figure(
                go.Pie(
                    labels=cb.index, values=cb["count"], hole=0.4,
                    marker=dict(colors=["steelblue", "tomato"]),
                    textinfo="label+percent",
                )
            )
            fig_cb.update_layout(height=380, title="income (<=50K vs >50K)")
            st.plotly_chart(fig_cb, width="stretch")

        # Valeurs manquantes par variable
        with g2:
            st.subheader("Valeurs manquantes")
            mv = analyzer.missing_values()
            if mv.empty:
                st.success("Aucune valeur manquante.")
            else:
                mv_df = mv.reset_index(names="variable")
                fig_mv = px.bar(
                    mv_df, x="pct", y="variable", orientation="h",
                    text=mv_df["pct"].map(lambda v: f"{v:.1f}%"),
                    color="pct", color_continuous_scale="Oranges",
                )
                fig_mv.update_layout(
                    height=380, title="Part de valeurs manquantes",
                    xaxis_title="% de lignes", yaxis_title="", coloraxis_showscale=False,
                )
                fig_mv.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_mv, width="stretch")

    # ------------------------------------------------------------------
    # Numériques : distribution + boxplot par classe (univarié/bivarié)
    # ------------------------------------------------------------------
    with sub_num:
        st.subheader("Distribution d'une variable numérique selon le revenu")
        col = st.selectbox("Variable", NUMERICAL_FEATURES, key="numcol")
        split_by_target = st.checkbox("Colorer par classe de revenu", value=True, key="numsplit")
        color = TARGET if split_by_target else None

        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(
                df_train, x=col, color=color, barmode="overlay", nbins=40,
                marginal="box", color_discrete_sequence=["steelblue", "tomato"],
            )
            fig.update_layout(height=460, title=f"Distribution de {col}")
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig_box = px.box(
                df_train, x=TARGET, y=col, color=TARGET,
                color_discrete_sequence=["steelblue", "tomato"],
            )
            fig_box.update_layout(height=460, title=f"{col} par classe de revenu",
                                  showlegend=False)
            st.plotly_chart(fig_box, width="stretch")

        st.subheader("Test de Mann-Whitney (numérique vs revenu)")
        st.caption("Effet (rank-biserial) > 0 ⇒ valeurs plus élevées chez les >50K. "
                   "À grand n, lire la taille d'effet plutôt que la p-value.")
        st.dataframe(analyzer.mann_whitney_target().round(4), width="stretch")

    # ------------------------------------------------------------------
    # Catégorielles : taux >50K par modalité (bivarié) + distribution
    # ------------------------------------------------------------------
    with sub_cat:
        st.subheader("Taux de >50K par modalité")
        catcol = st.selectbox("Variable catégorielle", CATEGORICAL_FEATURES, key="catcol")
        keep_na = st.checkbox("Garder les valeurs manquantes (Manquant)", value=True, key="catna")

        rate_tbl = analyzer.income_rate_by(catcol, dropna=not keep_na).reset_index()
        fig_rate = px.bar(
            rate_tbl, x="taux_>50K (%)", y="modalité", orientation="h",
            color="taux_>50K (%)", color_continuous_scale="RdYlGn",
            hover_data=["effectif"],
        )
        fig_rate.add_vline(
            x=baseline_rate, line_dash="dash", line_color="black",
            annotation_text=f"moyenne {baseline_rate:.1f}%", annotation_position="top",
        )
        fig_rate.update_layout(
            height=max(380, 26 * len(rate_tbl)),
            title=f"Taux de >50K selon {catcol}",
            yaxis_title="", coloraxis_showscale=False,
        )
        fig_rate.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_rate, width="stretch")
        st.caption("Une modalité au-dessus de la ligne sur-représente les hauts revenus, "
                   "en-dessous les sous-représente. Loin de la ligne = signal fort.")

    # ------------------------------------------------------------------
    # Corrélations : matrice numérique (Pearson) + Cramér's V (catégorielles)
    # ------------------------------------------------------------------
    with sub_corr:
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Corrélations numériques (Pearson)")
            corr = analyzer.X[NUMERICAL_FEATURES].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
            )
            fig_corr.update_layout(height=520)
            st.plotly_chart(fig_corr, width="stretch")
        with g2:
            st.subheader("Cramér's V (catégorielles + cible)")
            mat = cramers_matrix()
            fig_v = px.imshow(
                mat, text_auto=".2f", color_continuous_scale="Reds",
                zmin=0, zmax=1, aspect="auto",
            )
            fig_v.update_layout(height=520)
            st.plotly_chart(fig_v, width="stretch")
            st.caption("0 = indépendant, 1 = association parfaite. La colonne "
                       f"« {TARGET} » classe le lien catégorielle ↔ revenu.")

    # ------------------------------------------------------------------
    # Nuage de points (2 variables numériques)
    # ------------------------------------------------------------------
    with sub_scatter:
        st.subheader("Nuage de points (2 variables numériques)")
        cc1, cc2 = st.columns(2)
        xcol = cc1.selectbox("Axe X", NUMERICAL_FEATURES, index=0, key="scx")
        ycol = cc2.selectbox("Axe Y", NUMERICAL_FEATURES,
                             index=min(1, len(NUMERICAL_FEATURES) - 1), key="scy")
        sample = df_train.sample(min(3000, len(df_train)), random_state=0)
        fig_sc = px.scatter(
            sample, x=xcol, y=ycol, color=TARGET, opacity=0.6,
            color_discrete_sequence=["steelblue", "tomato"],
        )
        fig_sc.update_layout(height=560)
        st.plotly_chart(fig_sc, width="stretch")


# ======================================================================
# Onglet 2 — Modèle & hyperparamètres
# ======================================================================
with tab_model:
    st.header("Réaction du modèle aux hyperparamètres")

    st.sidebar.title("⚙️ Configuration")
    available = [m for m in ModelRegistry.available() if m in HYPERPARAMS]
    model_name = st.sidebar.selectbox("Modèle", available, index=available.index("random_forest"))

    # Source des hyperparamètres : manuels ou optimaux (Optuna) si dispo
    has_optuna = model_name in BEST_PARAMS
    source_options = ["Manuel"]
    if has_optuna:
        source_options.append("Optimaux (Optuna · AP)")
    source = st.sidebar.radio(
        "Hyperparamètres", source_options,
        index=len(source_options) - 1,  # défaut = optimaux quand disponibles
    )
    use_optuna = source.startswith("Optimaux")

    scaling = st.sidebar.checkbox("Scaling des variables numériques", value=True)
    st.sidebar.markdown("---")

    if use_optuna:
        entry = BEST_PARAMS[model_name]
        chosen = dict(entry["best_params"])
        st.sidebar.subheader("Hyperparamètres optimaux")
        st.sidebar.caption(
            f"Optuna · {entry.get('scoring', '?')} = {entry.get('best_value', float('nan')):.4f} "
            f"· {entry.get('n_trials', '?')} essais"
        )
        st.sidebar.json(chosen)
    else:
        st.sidebar.subheader("Hyperparamètres")
        # Widgets adaptés au modèle choisi (label nettoyé du préfixe set_params)
        chosen = {}
        for pname, (kind, kwargs) in HYPERPARAMS[model_name].items():
            chosen[pname] = build_widget(pname.split("__")[-1], kind, kwargs)

    run = st.sidebar.button("🚀 Entraîner / Évaluer", type="primary", width="stretch")

    st.write(f"**Modèle sélectionné :** `{model_name}`")
    st.write(f"**Source des hyperparamètres :** {source}")
    st.json(chosen, expanded=False)

    if run:
        with st.spinner("Entraînement en cours…"):
            # 1. Estimateur de base depuis le registre + hyperparamètres choisis
            params = dict(chosen)
            # liblinear est requis pour la pénalité L1 en régression logistique
            if model_name == "logistic_regression" and params.get("penalty") == "l1":
                params["solver"] = "liblinear"
            estimator = ModelRegistry.get(model_name).set_params(**params)

            # 2. Pipeline preprocessing + modèle
            pp = PreprocessingPipeline(scaling=scaling)
            y_tr = pp.fit_transform_target(y_train)
            y_te = pp.transform_target(y_test)
            pipeline = pp.build_pipeline(estimator)

            # 3. Évaluation sur le test set via ModelEvaluator
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_on_test(
                model_name, pipeline, X_train, y_tr, X_test, y_te
            )

            # Prédictions pour les graphes
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

        st.success("Terminé ✔")

        # --- Métriques ---
        cols = st.columns(len(metrics))
        for col_w, (k, v) in zip(cols, metrics.items()):
            col_w.metric(k, f"{v:.4f}")

        g1, g2 = st.columns(2)

        # --- Matrice de confusion (Plotly) ---
        with g1:
            cm = confusion_matrix(y_te, y_pred)
            labels = ["<=50K", ">50K"]
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=labels, y=labels, aspect="auto",
                labels=dict(x="Prédit", y="Réel", color="count"),
            )
            fig_cm.update_layout(title="Matrice de confusion", height=420)
            st.plotly_chart(fig_cm, width="stretch")

        # --- Courbe ROC (Plotly) ---
        with g2:
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            auc = roc_auc_score(y_te, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"ROC (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash"), name="Aléatoire"))
            fig_roc.update_layout(title="Courbe ROC", height=420,
                                  xaxis_title="Taux faux positifs",
                                  yaxis_title="Taux vrais positifs")
            st.plotly_chart(fig_roc, width="stretch")
    else:
        st.info("Choisis un modèle et des hyperparamètres dans la barre latérale, "
                "puis clique sur **Entraîner / Évaluer**.")
