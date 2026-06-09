import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo



class DataLoader:
    """Downloads and splits the Adult Income dataset (UCI id=2)."""

    DATASET_ID = 2

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        """Fetch dataset from UCI repository and return (X, y)."""
        dataset = fetch_ucirepo(id=self.DATASET_ID)
        X: pd.DataFrame = dataset.data.features.copy()
        y: pd.Series = dataset.data.targets.squeeze().copy()

        # Normalise target labels: strip whitespace and trailing dots
        y = y.str.strip().str.rstrip(".")

        return X, y

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
