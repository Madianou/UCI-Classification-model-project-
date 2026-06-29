from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import RAW_COLUMNS, TARGET


class DataLoader:
    """Loads the Adult Income dataset from the local UCI files.

    The raw files live in ``data/adult/`` :

    - ``adult.data`` : 32 561 training rows (labels ``<=50K`` / ``>50K``)
    - ``adult.test``  : 16 281 test rows. Its first line is a junk comment and
      its labels carry a trailing dot (``<=50K.``), both handled here.

    Only the **official split** is exposed: each file *is* a split. The test set
    is meant to stay untouched until final evaluation : in particular the EDA
    should run on the **train set only** (use :meth:`load_train`).
    """

    DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "adult"
    TRAIN_FILE = DATA_DIR / "adult.data"
    TEST_FILE = DATA_DIR / "adult.test"

    # ------------------------------------------------------------------
    # Raw file reading
    # ------------------------------------------------------------------

    def _read_file(self, path: Path, skiprows: int = 0) -> tuple[pd.DataFrame, pd.Series]:
        """Read one headerless Adult CSV and return cleaned (X, y).

        ``skipinitialspace`` strips the space after each comma, ``na_values``
        turns the ``?`` marker into real NaN, and trailing dots / whitespace are
        removed from the target labels.
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
        # Drop any residual blank/garbage rows (e.g. trailing newline).
        df = df.dropna(subset=[TARGET])

        y = df[TARGET].astype(str).str.strip().str.rstrip(".")
        X = df.drop(columns=[TARGET])
        # Belt-and-braces: normalise any stray "?" missed by na_values.
        X = X.replace("?", np.nan)
        return X.reset_index(drop=True), y.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_train(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X_train, y_train) from ``adult.data`` for EDA, preproc, modeling and opti."""
        return self._read_file(self.TRAIN_FILE)

    def load_test(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X_test, y_test) from ``adult.test`` for final evaluation only."""
        return self._read_file(self.TEST_FILE, skiprows=1)

    def load_and_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return (X_train, X_test, y_train, y_test) using the official split."""
        X_train, y_train = self.load_train()
        X_test, y_test = self.load_test()
        return X_train, X_test, y_train, y_test

    @staticmethod
    def class_distribution(y: pd.Series) -> pd.DataFrame:
        counts = y.value_counts()
        return pd.DataFrame({"count": counts, "pct": counts / len(y) * 100})
