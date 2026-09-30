"""Feature engineering helpers that keep post-release sales out of predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = ["Platform", "Genre", "Publisher", "Year"]
CATEGORICAL_FEATURES = ["Platform", "Genre", "Publisher"]


def add_console_generation(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a broad generation-era label derived from release year."""
    result = frame.copy()
    year = pd.to_numeric(result["Year"], errors="coerce")
    bins = [-np.inf, 1982, 1987, 1993, 1998, 2004, 2011, 2016, np.inf]
    labels = ["Gen 1", "Gen 2", "Gen 3", "Gen 4", "Gen 5", "Gen 6", "Gen 7", "Gen 8+"]
    result["Console_Generation"] = pd.cut(year, bins=bins, labels=labels).astype("string")
    return result


def add_sales_tier(frame: pd.DataFrame) -> pd.DataFrame:
    """Label titles by global-sales tertiles; tied quantile values stay together."""
    if "Global_Sales" not in frame:
        raise ValueError("Global_Sales is required to create Sales_Tier")
    result = frame.copy()
    sales = pd.to_numeric(result["Global_Sales"], errors="coerce")
    if sales.isna().any():
        raise ValueError("Global_Sales must be numeric and non-missing to create Sales_Tier")
    low_cutoff, high_cutoff = sales.quantile([1 / 3, 2 / 3]).to_numpy()
    result["Sales_Tier"] = np.select(
        [sales <= low_cutoff, sales <= high_cutoff],
        ["Low", "Medium"],
        default="High",
    )
    return result


def prepare_model_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return only pre-release predictors and the target, with stable types."""
    missing = sorted(set(MODEL_FEATURES + ["Sales_Tier"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Model data is missing required columns: {', '.join(missing)}")
    features = frame.loc[:, MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Unknown").astype(str)
    features["Year"] = pd.to_numeric(features["Year"], errors="coerce")
    if features["Year"].isna().any():
        features["Year"] = features["Year"].fillna(features["Year"].median())
    target = frame["Sales_Tier"].astype(str)
    return features, target
