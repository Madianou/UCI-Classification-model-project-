import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from ..data.schemas import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    POSITIVE_CLASS,
    TARGET,
)
from ..preprocessing.pipeline import PreprocessingPipeline


class ClusteringAnalyzer:
    """Analyse non-supervisée (t-SNE & k-means) du dataset Adult Income.

    Méthodes **non-supervisées** : la cible ``income`` n'entre jamais dans
    l'apprentissage, elle ne sert qu'à *colorier* et *profiler* les résultats.
    L'objet n'a donc pas vocation à modéliser, mais à explorer la structure des
    données et à illustrer le rapport (les classes se séparent-elles ? quels
    profils socio-économiques ressortent ?).

    Pipeline de référence (entrée commune aux deux méthodes) :
    ``X`` brut -> :class:`PreprocessingPipeline` (one-hot + scaling) -> matrice
    numérique ``Z`` -> PCA intermédiaire ``_reduced`` partagée par k-means
    (débruitage) et t-SNE (pré-réduction).

    Limite à garder en tête : après one-hot, l'espace est surtout binaire et la
    distance euclidienne — sur laquelle reposent k-means et t-SNE — y est un peu
    artificielle. La PCA intermédiaire atténue ce point sans le supprimer ; les
    conclusions restent **qualitatives**.

    Parameters
    ----------
    X, y : le train uniquement (cohérent avec l'AED).
    scaling : True par défaut, **indispensable** ici (distances euclidiennes).
    random_state : graine partagée (PCA, k-means, t-SNE, échantillonnage).
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        scaling: bool = True,
        random_state: int = 42,
    ) -> None:
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.random_state = random_state

        # Préprocessing : on réutilise la pipeline du projet. Pour une analyse
        # exploratoire non-supervisée, un fit_transform sur tout le train est
        # légitime (pas de cible apprise, donc pas de fuite à craindre).
        self._pipeline = PreprocessingPipeline(scaling=scaling)
        preprocessor = self._pipeline.build()
        self.Z = preprocessor.fit_transform(self.X)

        # 1 pour >50K, 0 sinon : pratique pour les taux et la coloration.
        self._is_pos = (self.y == POSITIVE_CLASS).astype(int).to_numpy()

        # Caches (calculs coûteux faits une seule fois).
        self._pca: PCA | None = None
        self._reduced: np.ndarray | None = None
        self._kmeans: KMeans | None = None
        self.labels_: np.ndarray | None = None
        self._tsne_idx: np.ndarray | None = None
        self._tsne_emb: np.ndarray | None = None

    # ------------------------------------------------------------------
    # PCA intermédiaire (représentation partagée k-means / t-SNE)
    # ------------------------------------------------------------------

    def reduce(self, n_components: int = 30) -> np.ndarray:
        """PCA sur la matrice scalée ; représentation partagée et mise en cache.

        ~30 composantes suffisent à débruiter le one-hot (>100 colonnes) tout en
        gardant l'essentiel de la variance. Recalcule si ``n_components`` change.
        """
        if self._reduced is None or self._pca is None or self._pca.n_components != n_components:
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            self._reduced = self._pca.fit_transform(self.Z)
        return self._reduced

    def explained_variance(self, n_components: int = 30) -> pd.DataFrame:
        """Variance expliquée (individuelle et cumulée) des composantes PCA."""
        self.reduce(n_components)
        ratio = self._pca.explained_variance_ratio_
        return pd.DataFrame(
            {
                "explained_variance": ratio,
                "cumulative": np.cumsum(ratio),
            },
            index=pd.RangeIndex(1, len(ratio) + 1, name="composante"),
        )

    def plot_explained_variance(
        self, n_components: int = 30, figsize: tuple = (9, 4)
    ) -> plt.Figure:
        """Éboulis (scree plot) : variance par composante + courbe cumulée."""
        ev = self.explained_variance(n_components)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(ev.index, ev["explained_variance"], color="steelblue",
               label="par composante")
        ax2 = ax.twinx()
        ax2.plot(ev.index, ev["cumulative"], "o-", color="tomato",
                 label="cumulée")
        ax2.set_ylim(0, 1.02)
        ax.set_xlabel("composante principale")
        ax.set_ylabel("variance expliquée")
        ax2.set_ylabel("variance cumulée")
        ax.set_title(f"PCA — éboulis des valeurs propres ({n_components} comp.)")
        fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=8)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # k-means
    # ------------------------------------------------------------------

    def kmeans_search(
        self,
        k_values: range | list[int] = range(2, 11),
        n_components: int = 30,
        silhouette_sample: int = 5000,
    ) -> pd.DataFrame:
        """Inertie et silhouette pour chaque k (choix du nombre de clusters).

        - inertie -> méthode du coude (chute puis aplatissement) ;
        - silhouette -> plus proche de 1 = clusters mieux séparés.

        k-means scale bien sur ~32k lignes, on ne sous-échantillonne donc PAS
        l'entraînement. En revanche la silhouette est en O(n²) : on l'estime sur
        ``silhouette_sample`` points (argument natif de sklearn) pour rester
        rapide. k-means++ et plusieurs ``n_init`` couvrent la sensibilité à
        l'initialisation.
        """
        reduced = self.reduce(n_components)
        rows = []
        for k in k_values:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                        random_state=self.random_state)
            labels = km.fit_predict(reduced)
            sil = silhouette_score(
                reduced, labels,
                sample_size=min(silhouette_sample, len(reduced)),
                random_state=self.random_state,
            )
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        return pd.DataFrame(rows).set_index("k")

    def plot_kmeans_search(
        self,
        k_values: range | list[int] = range(2, 11),
        n_components: int = 30,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Coude (inertie) et silhouette côte à côte pour choisir k."""
        scores = self.kmeans_search(k_values, n_components)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.plot(scores.index, scores["inertia"], "o-", color="steelblue")
        ax1.set_xlabel("k")
        ax1.set_ylabel("inertie")
        ax1.set_title("Méthode du coude")
        best = scores["silhouette"].idxmax()
        ax2.plot(scores.index, scores["silhouette"], "o-", color="tomato")
        ax2.axvline(best, ls="--", color="grey", lw=1,
                    label=f"max à k={best}")
        ax2.set_xlabel("k")
        ax2.set_ylabel("silhouette")
        ax2.set_title("Score de silhouette")
        ax2.legend(fontsize=8)
        fig.suptitle("Choix du nombre de clusters", fontweight="bold")
        fig.tight_layout()
        return fig

    def fit_kmeans(self, k: int, n_components: int = 30) -> np.ndarray:
        """Ajuste le k-means final et mémorise les labels (``labels_``)."""
        reduced = self.reduce(n_components)
        self._kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10,
                              random_state=self.random_state)
        self.labels_ = self._kmeans.fit_predict(reduced)
        return self.labels_

    def cluster_profile(
        self,
        numeric: list[str] | None = None,
        extra: dict[str, np.ndarray | pd.Series] | None = None,
    ) -> pd.DataFrame:
        """Profil interprétable de chaque cluster (après :meth:`fit_kmeans`).

        Par cluster : effectif, part du total, **taux de >50K** (à comparer au
        ~24 % global : un écart marqué = profil discriminant), moyenne des
        variables numériques choisies, et modalité dominante des catégorielles
        les plus parlantes. Construit sur ``X`` *brut* (pas la matrice scalée)
        pour rester lisible.

        ``extra`` injecte des colonnes additionnelles **moyennées par cluster**
        à partir de vecteurs alignés positionnellement sur ``X`` (longueur =
        ``len(self.X)``) — typiquement, en évaluation, le **taux d'erreur du
        modèle** pour repérer les clusters « frontière » qui concentrent les
        mauvaises prédictions.
        """
        if self.labels_ is None:
            raise RuntimeError("Appeler fit_kmeans(k) avant cluster_profile().")
        numeric = numeric or NUMERICAL_FEATURES
        prof = self.X.copy()
        prof["cluster"] = self.labels_
        prof["_pos"] = self._is_pos

        g = prof.groupby("cluster")
        out = pd.DataFrame({
            "effectif": g.size(),
            "pct": g.size() / len(prof) * 100,
            "taux_>50K (%)": g["_pos"].mean() * 100,
        })
        for name, vec in (extra or {}).items():
            out[name] = (
                pd.Series(np.asarray(vec), index=prof.index)
                .groupby(self.labels_).mean()
            )
        for col in numeric:
            out[f"{col} (moy.)"] = g[col].mean()
        # Modalité dominante des catégorielles les plus interprétables.
        for col in ["occupation", "marital-status", "education", "sex"]:
            if col in CATEGORICAL_FEATURES:
                out[f"{col} (top)"] = g[col].agg(
                    lambda s: s.mode().iat[0] if not s.mode().empty else np.nan
                )
        return out.sort_values("taux_>50K (%)", ascending=False)

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------

    def tsne_embedding(
        self,
        sample_size: int = 6000,
        perplexity: float = 30.0,
        n_components: int = 30,
    ) -> np.ndarray:
        """Projection 2D t-SNE sur un échantillon stratifié (mise en cache).

        t-SNE est en ~O(n log n) mais reste lourd sur ~32k lignes : on travaille
        sur un échantillon stratifié par ``income`` (~6k points) qui préserve le
        ratio de classes. On part de la représentation PCA (pré-réduction
        recommandée au-delà de ~50 dimensions). ``perplexity`` (~5–50) influence
        beaucoup le rendu, à explorer. ``n_jobs=-1`` exploite les cœurs locaux.

        Rappel : sur un t-SNE, **les distances entre clusters et leurs tailles
        ne sont pas fiables** — lecture purement qualitative.
        """
        reduced = self.reduce(n_components)
        idx = np.arange(len(reduced))
        if sample_size < len(idx):
            idx, _ = train_test_split(
                idx, train_size=sample_size, stratify=self.y,
                random_state=self.random_state,
            )
        tsne = TSNE(
            n_components=2, perplexity=perplexity,
            init="pca", random_state=self.random_state, n_jobs=-1,
        )
        self._tsne_idx = idx
        self._tsne_emb = tsne.fit_transform(reduced[idx])
        return self._tsne_emb

    def plot_tsne(
        self,
        color_by: str = "income",
        sample_size: int = 6000,
        perplexity: float = 30.0,
        figsize: tuple = (8, 7),
        *,
        values: np.ndarray | pd.Series | None = None,
        value_labels: dict | None = None,
        value_colors: dict | None = None,
        legend_title: str | None = None,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Nuage t-SNE 2D colorié par ``income``, par cluster ou par un vecteur.

        - ``color_by="income"`` : les deux classes forment-elles des régions
          séparées ? (lecture directe de la difficulté du problème de classif) ;
        - ``color_by="cluster"`` : validation croisée — si les clusters k-means
          coïncident avec des régions cohérentes du t-SNE, ils captent une vraie
          structure. Nécessite :meth:`fit_kmeans` au préalable ;
        - ``color_by="custom"`` : colorie par ``values`` (vecteur **aligné
          positionnellement** sur ``X``, longueur = ``len(self.X)``). Sert, en
          phase d'évaluation, à projeter les **prédictions** ou les **erreurs**
          du modèle sur la même carte t-SNE que les vraies classes.
          ``value_labels`` / ``value_colors`` (dicts indexés par valeur)
          personnalisent la légende et les couleurs.

        Réutilise l'embedding en cache s'il correspond au même ``sample_size``.
        Passer ``ax`` pour dessiner dans une grille de sous-graphes existante.
        """
        if (self._tsne_emb is None or self._tsne_idx is None
                or len(self._tsne_idx) != min(sample_size, len(self.Z))):
            self.tsne_embedding(sample_size, perplexity)
        emb, idx = self._tsne_emb, self._tsne_idx

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        if color_by == "income":
            for label, color in [("<=50K", "steelblue"), (">50K", "tomato")]:
                m = self.y.to_numpy()[idx] == label
                ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5,
                           color=color, label=label)
            ax.legend(title=legend_title or TARGET, markerscale=2)
            title = title or "t-SNE colorié par income"
        elif color_by == "cluster":
            if self.labels_ is None:
                raise RuntimeError("Appeler fit_kmeans(k) avant color_by='cluster'.")
            labels = self.labels_[idx]
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.5,
                            c=labels, cmap="tab10")
            ax.legend(*sc.legend_elements(), title=legend_title or "cluster",
                      markerscale=2)
            title = title or "t-SNE colorié par cluster k-means"
        elif color_by == "custom":
            if values is None:
                raise ValueError("color_by='custom' nécessite l'argument values.")
            v = np.asarray(values)[idx]
            palette = ["steelblue", "tomato", "seagreen", "goldenrod", "purple"]
            for j, u in enumerate(np.unique(v)):
                m = v == u
                label = value_labels.get(u, str(u)) if value_labels else str(u)
                color = (value_colors.get(u) if value_colors
                         else palette[j % len(palette)])
                ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5,
                           color=color, label=label)
            ax.legend(title=legend_title, markerscale=2)
            title = title or "t-SNE colorié (vecteur fourni)"
        else:
            raise ValueError("color_by doit être 'income', 'cluster' ou 'custom'")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"{title} (n={len(idx):,}, perplexity={perplexity:g})")
        fig.tight_layout()
        return fig
