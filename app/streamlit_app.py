"""
Démo interactive : Adult Income (UCI)

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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from adult_income import (
    ClusteringAnalyzer,
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

st.set_page_config(page_title="Adult Income : Démo", layout="wide")

# Fichier des hyperparamètres optimaux calculés par Optuna (scoring = AP).
BEST_PARAMS_FILE = ROOT / "artifacts" / "optuna" / "best_params.json"


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
# Analyse non-supervisée (t-SNE & k-means) : réutilise ClusteringAnalyzer.
#   Deux analyseurs distincts, chacun mis en cache une seule fois (objet
#   lourd : prétraitement + PCA au démarrage) :
#     - "train" : exploration de la structure des données (onglet 1) ;
#     - "test"  : projection des erreurs du modèle entraîné (onglet 2).
#   Les calculs coûteux (t-SNE, recherche de k) sont mis en cache par
#   paramètres et ne tournent que sur action de l'utilisateur (bouton).
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Préparation de l'analyse non-supervisée (train)…")
def clustering_train() -> ClusteringAnalyzer:
    return ClusteringAnalyzer(X_train, y_train)


@st.cache_resource(show_spinner="Préparation de l'analyse non-supervisée (test)…")
def clustering_test() -> ClusteringAnalyzer:
    return ClusteringAnalyzer(X_test, y_test)


def _clustering(which: str) -> ClusteringAnalyzer:
    return clustering_train() if which == "train" else clustering_test()


@st.cache_data(show_spinner="Recherche du nombre de clusters (k-means)…")
def kmeans_search(which: str, k_min: int, k_max: int) -> pd.DataFrame:
    return _clustering(which).kmeans_search(range(k_min, k_max + 1))


@st.cache_data(show_spinner="Projection t-SNE (peut prendre ~30 s)…")
def tsne_projection(which: str, sample_size: int, perplexity: float):
    ca = _clustering(which)
    emb = ca.tsne_embedding(sample_size=sample_size, perplexity=perplexity)
    return emb, ca._tsne_idx


@st.cache_data(show_spinner="k-means (clusters + profils)…")
def kmeans_fit(which: str, k: int):
    """Ajuste le k-means final et renvoie (labels, profil interprétable)."""
    ca = _clustering(which)
    labels = ca.fit_kmeans(k)
    return labels, ca.cluster_profile()


INCOME_COLORS = {"<=50K": "steelblue", ">50K": "tomato"}


def plot_kmeans_search(scores: pd.DataFrame) -> go.Figure:
    """Coude (inertie) + silhouette côte à côte, version Plotly."""
    best = int(scores["silhouette"].idxmax())
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Méthode du coude (inertie)", "Score de silhouette"),
    )
    fig.add_trace(
        go.Scatter(x=scores.index, y=scores["inertia"], mode="lines+markers",
                   line=dict(color="steelblue")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=scores.index, y=scores["silhouette"], mode="lines+markers",
                   line=dict(color="tomato")),
        row=1, col=2,
    )
    fig.add_vline(x=best, line_dash="dash", line_color="grey", row=1, col=2)
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_layout(
        height=380, showlegend=False,
        title=f"Choix du nombre de clusters : silhouette maximale à k={best}",
    )
    return fig


def plot_tsne(df: pd.DataFrame, color_col: str, title: str,
              color_map: dict | None = None) -> go.Figure:
    """Nuage t-SNE 2D Plotly (axes masqués : seules les positions relatives comptent)."""
    fig = px.scatter(
        df, x="x", y="y", color=color_col, opacity=0.6,
        color_discrete_map=color_map,
        hover_data={"x": False, "y": False},
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=460, title=title,
                      xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                      legend_title=color_col)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ----------------------------------------------------------------------
# Spécification des hyperparamètres exposés par modèle (mode manuel)
#   chaque entrée : (type, kwargs du widget)  -> construit dans la sidebar
#   Les clés sont les NOMS RÉELS passés à set_params (knn préfixé "knn__"
#   car le registre renvoie un Pipeline PCA + KNN).
# ----------------------------------------------------------------------
HYPERPARAMS = {
    "logistic_regression": {
        "C": (
            "slider_float",
            dict(min_value=0.01, max_value=10.0, value=1.0, step=0.01),
        ),
        "penalty": ("select", dict(options=["l2", "l1"], index=0)),
    },
    "decision_tree": {
        "max_depth": ("slider_int", dict(min_value=1, max_value=30, value=10)),
        "min_samples_split": ("slider_int", dict(min_value=2, max_value=50, value=2)),
        "criterion": ("select", dict(options=["gini", "entropy"], index=0)),
    },
    "random_forest": {
        "n_estimators": (
            "slider_int",
            dict(min_value=10, max_value=500, value=100, step=10),
        ),
        "max_depth": ("slider_int", dict(min_value=1, max_value=40, value=20)),
        "min_samples_split": ("slider_int", dict(min_value=2, max_value=50, value=2)),
    },
    "gradient_boosting": {
        "n_estimators": (
            "slider_int",
            dict(min_value=10, max_value=500, value=100, step=10),
        ),
        "learning_rate": (
            "slider_float",
            dict(min_value=0.01, max_value=1.0, value=0.1, step=0.01),
        ),
        "max_depth": ("slider_int", dict(min_value=1, max_value=10, value=3)),
    },
    "xgboost": {
        "n_estimators": (
            "slider_int",
            dict(min_value=10, max_value=500, value=100, step=10),
        ),
        "learning_rate": (
            "slider_float",
            dict(min_value=0.01, max_value=1.0, value=0.3, step=0.01),
        ),
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
tab_data, tab_model = st.tabs([" Jeu de données", " Modèle & hyperparamètres"])


# ======================================================================
# Onglet 1 : Exploration du jeu de données (style EDA)
# ======================================================================
with tab_data:
    st.header("Jeu de données : Adult Income (UCI)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (train)", f"{len(df_train):,}")
    c2.metric("Variables", f"{X_train.shape[1]}")
    c3.metric("Taux >50K", f"{baseline_rate:.1f} %")
    c4.metric("Valeurs manquantes", f"{int(X_train.isnull().sum().sum()):,}")

    (
        sub_overview,
        sub_num,
        sub_cat,
        sub_corr,
        sub_scatter,
        sub_cluster,
    ) = st.tabs(
        [
            "Vue d'ensemble",
            "Numériques",
            "Catégorielles",
            "Corrélations",
            "Nuage de points",
            "Clustering (t-SNE & k-means)",
        ]
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
                    labels=cb.index,
                    values=cb["count"],
                    hole=0.4,
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
                    mv_df,
                    x="pct",
                    y="variable",
                    orientation="h",
                    text=mv_df["pct"].map(lambda v: f"{v:.1f}%"),
                    color="pct",
                    color_continuous_scale="Oranges",
                )
                fig_mv.update_layout(
                    height=380,
                    title="Part de valeurs manquantes",
                    xaxis_title="% de lignes",
                    yaxis_title="",
                    coloraxis_showscale=False,
                )
                fig_mv.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_mv, width="stretch")

    # ------------------------------------------------------------------
    # Numériques : distribution + boxplot par classe (univarié/bivarié)
    # ------------------------------------------------------------------
    with sub_num:
        st.subheader("Distribution d'une variable numérique selon le revenu")
        col = st.selectbox("Variable", NUMERICAL_FEATURES, key="numcol")
        split_by_target = st.checkbox(
            "Colorer par classe de revenu", value=True, key="numsplit"
        )
        color = TARGET if split_by_target else None

        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(
                df_train,
                x=col,
                color=color,
                barmode="overlay",
                nbins=40,
                marginal="box",
                color_discrete_sequence=["steelblue", "tomato"],
            )
            fig.update_layout(height=460, title=f"Distribution de {col}")
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig_box = px.box(
                df_train,
                x=TARGET,
                y=col,
                color=TARGET,
                color_discrete_sequence=["steelblue", "tomato"],
            )
            fig_box.update_layout(
                height=460, title=f"{col} par classe de revenu", showlegend=False
            )
            st.plotly_chart(fig_box, width="stretch")

    # ------------------------------------------------------------------
    # Catégorielles : taux >50K par modalité (bivarié) + distribution
    # ------------------------------------------------------------------
    with sub_cat:
        st.subheader("Taux de >50K par modalité")
        catcol = st.selectbox(
            "Variable catégorielle", CATEGORICAL_FEATURES, key="catcol"
        )
        keep_na = st.checkbox(
            "Garder les valeurs manquantes (Manquant)", value=True, key="catna"
        )

        rate_tbl = analyzer.income_rate_by(catcol, dropna=not keep_na).reset_index()
        fig_rate = px.bar(
            rate_tbl,
            x="taux_>50K (%)",
            y="modalité",
            orientation="h",
            color="taux_>50K (%)",
            color_continuous_scale="RdYlGn",
            hover_data=["effectif"],
        )
        fig_rate.add_vline(
            x=baseline_rate,
            line_dash="dash",
            line_color="black",
            annotation_text=f"moyenne {baseline_rate:.1f}%",
            annotation_position="top",
        )
        fig_rate.update_layout(
            height=max(380, 26 * len(rate_tbl)),
            title=f"Taux de >50K selon {catcol}",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        fig_rate.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_rate, width="stretch")
        st.caption(
            "Une modalité au-dessus de la ligne sur-représente les hauts revenus, "
            "en-dessous les sous-représente. Loin de la ligne = signal fort."
        )

    # ------------------------------------------------------------------
    # Corrélations : matrice numérique (Pearson) + Cramér's V (catégorielles)
    # ------------------------------------------------------------------
    with sub_corr:
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Corrélations numériques (Pearson)")
            corr = analyzer.X[NUMERICAL_FEATURES].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            fig_corr.update_layout(height=520)
            st.plotly_chart(fig_corr, width="stretch")
        with g2:
            st.subheader("Cramér's V (catégorielles + cible)")
            mat = cramers_matrix()
            fig_v = px.imshow(
                mat,
                text_auto=".2f",
                color_continuous_scale="Reds",
                zmin=0,
                zmax=1,
                aspect="auto",
            )
            fig_v.update_layout(height=520)
            st.plotly_chart(fig_v, width="stretch")
            st.caption(
                "0 = indépendant, 1 = association parfaite. La colonne "
                f"« {TARGET} » classe le lien catégorielle ↔ revenu."
            )

    # ------------------------------------------------------------------
    # Nuage de points (2 variables numériques)
    # ------------------------------------------------------------------
    with sub_scatter:
        st.subheader("Nuage de points (2 variables numériques)")
        cc1, cc2 = st.columns(2)
        xcol = cc1.selectbox("Axe X", NUMERICAL_FEATURES, index=0, key="scx")
        ycol = cc2.selectbox(
            "Axe Y",
            NUMERICAL_FEATURES,
            index=min(1, len(NUMERICAL_FEATURES) - 1),
            key="scy",
        )
        sample = df_train.sample(min(3000, len(df_train)), random_state=0)
        fig_sc = px.scatter(
            sample,
            x=xcol,
            y=ycol,
            color=TARGET,
            opacity=0.6,
            color_discrete_sequence=["steelblue", "tomato"],
        )
        fig_sc.update_layout(height=560)
        st.plotly_chart(fig_sc, width="stretch")

    # ------------------------------------------------------------------
    # Clustering : analyse non-supervisée (t-SNE & k-means) sur le train
    # ------------------------------------------------------------------
    with sub_cluster:
        st.subheader("Structure non-supervisée des données (t-SNE & k-means)")
        st.caption(
            "Analyse **non-supervisée** : le revenu n'entre jamais dans "
            "l'apprentissage, il ne sert qu'à colorier et profiler. "
            "t-SNE et k-means partagent une PCA intermédiaire (one-hot + scaling). "
            "Lecture **qualitative** : sur un t-SNE, tailles et distances entre "
            "groupes ne sont pas fiables."
        )

        cc1, cc2, cc3 = st.columns(3)
        k_clusters = cc1.slider("Nombre de clusters (k)", 2, 10, 4, key="cl_k")
        perplexity = cc2.slider("Perplexité (t-SNE)", 5, 50, 30, key="cl_perp")
        sample_size = cc3.slider(
            "Échantillon t-SNE", 1000, 10000, 5000, step=1000, key="cl_samp"
        )

        if st.button("Lancer l'analyse non-supervisée", type="primary", key="cl_run"):
            st.session_state["clustering_done"] = True

        if st.session_state.get("clustering_done"):
            # 1. Choix de k : coude + silhouette
            st.markdown("#### Choix du nombre de clusters")
            scores = kmeans_search("train", 2, 10)
            st.plotly_chart(plot_kmeans_search(scores), width="stretch")

            # 2. Projection t-SNE (coloriée par revenu, puis par cluster)
            labels, profile = kmeans_fit("train", k_clusters)
            emb, idx = tsne_projection("train", sample_size, perplexity)
            y_arr = y_train.reset_index(drop=True).to_numpy()
            df_tsne = pd.DataFrame({
                "x": emb[:, 0],
                "y": emb[:, 1],
                "income": y_arr[idx],
                "cluster": labels[idx].astype(str),
            })

            st.markdown("#### Projection t-SNE 2D")
            t1, t2 = st.columns(2)
            with t1:
                st.plotly_chart(
                    plot_tsne(df_tsne, "income", "Colorié par revenu", INCOME_COLORS),
                    width="stretch",
                )
                st.caption(
                    "Les classes se chevauchent largement → problème de "
                    "classification difficile, pas de frontière nette."
                )
            with t2:
                st.plotly_chart(
                    plot_tsne(
                        df_tsne.sort_values("cluster"),
                        "cluster",
                        f"Colorié par cluster k-means (k={k_clusters})",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Si les clusters k-means coïncident avec des régions cohérentes "
                    "du t-SNE, ils captent une vraie structure."
                )

            # 3. Profil interprétable des clusters
            st.markdown(f"#### Profil des {k_clusters} clusters")
            st.caption(
                f"Trié par taux de >50K (~{baseline_rate:.0f} % en moyenne) : "
                "un écart marqué = profil socio-économique discriminant."
            )
            st.dataframe(
                profile.style.format(precision=1).background_gradient(
                    subset=["taux_>50K (%)"], cmap="RdYlGn"
                ),
                width="stretch",
            )
        else:
            st.info(
                "Règle k / perplexité / échantillon puis clique sur "
                "**Lancer l'analyse non-supervisée** "
                "(le calcul t-SNE peut prendre ~30 s)."
            )


# ======================================================================
# Onglet 2 : Modèle & hyperparamètres
# ======================================================================
with tab_model:
    st.header("Réaction du modèle aux hyperparamètres")

    st.sidebar.title("Configuration")
    available = [m for m in ModelRegistry.available() if m in HYPERPARAMS]
    model_name = st.sidebar.selectbox(
        "Modèle", available, index=available.index("random_forest")
    )

    # Source des hyperparamètres : manuels ou optimaux (Optuna) si dispo
    has_optuna = model_name in BEST_PARAMS
    source_options = ["Manuel"]
    if has_optuna:
        source_options.append("Optimaux (Optuna · AP)")
    source = st.sidebar.radio(
        "Hyperparamètres",
        source_options,
        index=len(source_options) - 1,  # défaut = optimaux quand disponibles
    )
    use_optuna = source.startswith("Optimaux")

    scaling = st.sidebar.checkbox("Scaling des variables numériques", value=True)
    eval_tsne = st.sidebar.checkbox(
        "Projeter les erreurs sur une carte t-SNE (lent)", value=False
    )
    eval_k = st.sidebar.slider(
        "Clusters pour le profil d'erreurs (k)", 2, 10, 4, disabled=not eval_tsne
    )
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

    run = st.sidebar.button("Entraîner / Évaluer", type="primary", width="stretch")

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

        # --- Métriques (libellés lisibles ; AP = aire sous la courbe PR) ---
        metric_labels = {
            "accuracy": "Exactitude",
            "precision": "Précision",
            "recall": "Rappel",
            "f1": "F1",
            "roc_auc": "ROC AUC",
            "average_precision": "PR AUC (AP)",
        }
        cols = st.columns(len(metrics))
        for col_w, (k, v) in zip(cols, metrics.items()):
            col_w.metric(metric_labels.get(k, k), f"{v:.4f}")

        g1, g2 = st.columns(2)

        # --- Matrice de confusion (Plotly) ---
        with g1:
            cm = confusion_matrix(y_te, y_pred)
            labels = ["<=50K", ">50K"]
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                x=labels,
                y=labels,
                aspect="auto",
                labels=dict(x="Prédit", y="Réel", color="count"),
            )
            fig_cm.update_layout(title="Matrice de confusion", height=420)
            st.plotly_chart(fig_cm, width="stretch")

        # --- Courbe ROC (Plotly) ---
        with g2:
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            auc = roc_auc_score(y_te, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})")
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash"),
                    name="Aléatoire",
                )
            )
            fig_roc.update_layout(
                title="Courbe ROC",
                height=420,
                xaxis_title="Taux faux positifs",
                yaxis_title="Taux vrais positifs",
            )
            st.plotly_chart(fig_roc, width="stretch")

        g3, g4 = st.columns(2)

        # --- Courbe Précision-Rappel (PR AUC = Average Precision) ---
        with g3:
            prec, rec, _ = precision_recall_curve(y_te, y_proba)
            ap = average_precision_score(y_te, y_proba)
            baseline_pr = float(np.mean(np.asarray(y_te) == 1))
            fig_pr = go.Figure()
            fig_pr.add_trace(
                go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP={ap:.3f})")
            )
            fig_pr.add_hline(
                y=baseline_pr,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"Aléatoire (AP={baseline_pr:.2f})",
                annotation_position="bottom right",
            )
            fig_pr.update_layout(
                title="Courbe Précision-Rappel (PR AUC = AP)",
                height=420,
                xaxis_title="Rappel",
                yaxis_title="Précision",
            )
            st.plotly_chart(fig_pr, width="stretch")

        with g4:
            st.markdown("##### PR AUC vs ROC AUC")
            st.markdown(
                "Sur une cible **déséquilibrée** (~24 % de >50K), la courbe "
                "Précision-Rappel est plus informative que la ROC : elle se "
                "concentre sur la classe positive minoritaire et son aire (AP) "
                "chute dès que les faux positifs augmentent.\n\n"
                f"- **PR AUC (AP)** = `{ap:.4f}` : base aléatoire ≈ "
                f"`{baseline_pr:.2f}`\n"
                f"- **ROC AUC** = `{auc:.4f}` : base aléatoire = `0.50`"
            )

        # --- Projection t-SNE des erreurs du modèle (jeu de test) ---
        if eval_tsne:
            st.markdown("---")
            st.subheader("Où le modèle se trompe-t-il ? (t-SNE du jeu de test)")
            st.caption(
                "Carte non-supervisée du jeu de **test** (perplexité=30, "
                "échantillon=5000). On y projette les prédictions et les erreurs "
                "du modèle entraîné pour localiser les zones difficiles."
            )

            y_pred_arr = np.asarray(y_pred)
            y_true_arr = np.asarray(y_te)
            err = (y_pred_arr != y_true_arr).astype(int)  # 1 = erreur

            ca_test = clustering_test()
            emb_t, idx_t = tsne_projection("test", 5000, 30.0)
            y_test_str = y_test.reset_index(drop=True).to_numpy()
            df_eval = pd.DataFrame({
                "x": emb_t[:, 0],
                "y": emb_t[:, 1],
                "Vraie classe": y_test_str[idx_t],
                "Prédiction": np.where(y_pred_arr[idx_t] == 1, ">50K", "<=50K"),
                "Résultat": np.where(err[idx_t] == 1, "Erreur", "Correct"),
            })

            e1, e2, e3 = st.columns(3)
            with e1:
                st.plotly_chart(
                    plot_tsne(df_eval, "Vraie classe", "Vraie classe", INCOME_COLORS),
                    width="stretch",
                )
            with e2:
                st.plotly_chart(
                    plot_tsne(df_eval, "Prédiction", "Prédiction du modèle",
                              INCOME_COLORS),
                    width="stretch",
                )
            with e3:
                st.plotly_chart(
                    plot_tsne(
                        df_eval, "Résultat", "Erreurs du modèle",
                        {"Correct": "lightgrey", "Erreur": "crimson"},
                    ),
                    width="stretch",
                )

            # Taux d'erreur par cluster k-means (profil des zones difficiles)
            st.markdown(f"##### Taux d'erreur par cluster (k={eval_k})")
            st.caption(
                "Les clusters au taux d'erreur élevé sont les profils "
                "« frontière » que le modèle confond le plus."
            )
            ca_test.fit_kmeans(eval_k)
            profile_err = ca_test.cluster_profile(
                extra={"taux_erreur (%)": err * 100}
            )
            st.dataframe(
                profile_err.style.format(precision=1).background_gradient(
                    subset=["taux_erreur (%)"], cmap="Reds"
                ),
                width="stretch",
            )
    else:
        st.info(
            "Choisis un modèle et des hyperparamètres dans la barre latérale, "
            "puis clique sur **Entraîner / Évaluer**."
        )
