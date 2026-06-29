from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import RAW_COLUMNS, TARGET


class DataLoader:
    """Charge le dataset Adult Income depuis les fichiers UCI locaux.

    Les fichiers bruts sont dans ``data/adult/`` :

    - ``adult.data`` : 32 561 lignes d'entraînement (labels ``<=50K`` / ``>50K``)
    - ``adult.test``  : 16 281 lignes de test. Sa 1re ligne est un commentaire
      parasite et ses labels ont un point final (``<=50K.``), gérés ici.

    Seul le **split officiel** est exposé : un fichier = un split. Le test reste
    intact jusqu'à l'évaluation finale ; en particulier l'AED tourne sur le
    **train seul** (voir :meth:`load_train`).
    """

    DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "adult"
    TRAIN_FILE = DATA_DIR / "adult.data"
    TEST_FILE = DATA_DIR / "adult.test"

    # ------------------------------------------------------------------
    # Lecture des fichiers bruts
    # ------------------------------------------------------------------

    def _read_file(self, path: Path, skiprows: int = 0) -> tuple[pd.DataFrame, pd.Series]:
        """Lit un CSV Adult sans en-tête et renvoie (X, y) nettoyés.

        ``skipinitialspace`` retire l'espace après chaque virgule, ``na_values``
        transforme le marqueur ``?`` en vrai NaN, et les points/espaces finaux
        sont retirés des labels de la cible.
        """
        df = pd.read_csv(
            path,
            header=None,
            names=RAW_COLUMNS,
            skipinitialspace=True,
            skiprows=skiprows,
            na_values="?",
            skip_blank_lines=True,
        )
        # Retire les lignes vides/parasites résiduelles (ex. saut de ligne final).
        df = df.dropna(subset=[TARGET])

        y = df[TARGET].astype(str).str.strip().str.rstrip(".")
        X = df.drop(columns=[TARGET])
        # Par sécurité : normalise tout "?" qu'aurait manqué na_values.
        X = X.replace("?", np.nan)
        return X.reset_index(drop=True), y.reset_index(drop=True)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def load_train(self) -> tuple[pd.DataFrame, pd.Series]:
        """(X_train, y_train) depuis ``adult.data`` : AED, preproc, modélisation, opti."""
        return self._read_file(self.TRAIN_FILE)

    def load_test(self) -> tuple[pd.DataFrame, pd.Series]:
        """(X_test, y_test) depuis ``adult.test`` : évaluation finale uniquement."""
        return self._read_file(self.TEST_FILE, skiprows=1)

    def load_and_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """(X_train, X_test, y_train, y_test) selon le split officiel."""
        X_train, y_train = self.load_train()
        X_test, y_test = self.load_test()
        return X_train, X_test, y_train, y_test

    @staticmethod
    def class_distribution(y: pd.Series) -> pd.DataFrame:
        """Effectif et pourcentage de chaque classe de la cible."""
        counts = y.value_counts()
        return pd.DataFrame({"count": counts, "pct": counts / len(y) * 100})
