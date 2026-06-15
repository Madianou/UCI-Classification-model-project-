from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from .schemas import TARGET


class DataLoader:
    """Downloads and splits the Adult Income dataset (UCI id=2)."""

    DATASET_ID = 2
    # Project-root/data/raw/adult.pkl (loader.py is src/adult_income/data/).
    # Pickle keeps dtypes/NaN exactly and needs no extra engine (unlike parquet).
    CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "adult.pkl"

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        use_cache: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.use_cache = use_cache

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y), served from a local parquet cache when available.

        The first call fetches from UCI (slow) and caches the cleaned data;
        subsequent calls read the local file instantly. Set use_cache=False
        or call clear_cache() to force a fresh download.
        """
        if self.use_cache and self.CACHE_PATH.exists():
            df = pd.read_pickle(self.CACHE_PATH)
            return df.drop(columns=[TARGET]), df[TARGET]

        dataset = fetch_ucirepo(id=self.DATASET_ID)
        X: pd.DataFrame = dataset.data.features.copy()
        y: pd.Series = dataset.data.targets.squeeze().copy()

        # The Adult dataset uses "?" as a missing-value marker in part of the
        # data. Normalise it to real NaN at the source so every downstream
        # consumer (EDA, preprocessing) works with a single missing convention.
        X = X.replace("?", np.nan)

        # Normalise target labels: strip whitespace and trailing dots
        y = y.str.strip().str.rstrip(".")

        if self.use_cache:
            self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df = X.copy()
            df[TARGET] = y.values
            df.to_pickle(self.CACHE_PATH)

        return X, y

    @classmethod
    def clear_cache(cls) -> None:
        """Delete the local parquet cache so the next load() refetches."""
        cls.CACHE_PATH.unlink(missing_ok=True)

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train/test split."""
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def load_and_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Convenience: load then split in one call."""
        X, y = self.load()
        return self.split(X, y)

    @staticmethod
    def class_distribution(y: pd.Series) -> pd.DataFrame:
        counts = y.value_counts()
        return pd.DataFrame({"count": counts, "pct": counts / len(y) * 100})
