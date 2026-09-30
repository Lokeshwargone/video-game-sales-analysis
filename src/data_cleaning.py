"""Data loading and cleaning helpers for the video game sales dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SALES_COLUMNS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
REQUIRED_COLUMNS = ["Name", "Platform", "Year", "Genre", "Publisher", *SALES_COLUMNS]


def load_sales_data(path: str | Path) -> pd.DataFrame:
    """Read the source CSV and validate its expected schema."""
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return frame


def clean_sales_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize labels, coerce data types, impute key fields, and remove duplicates.

    Year values that cannot be parsed are imputed with the median observed year.
    Missing publishers use an explicit ``Unknown`` category. Rows without a game
    name, platform, or genre are excluded because they cannot be analyzed reliably.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    cleaned = frame.loc[:, REQUIRED_COLUMNS].copy()
    rows_before = len(cleaned)
    cleaned["Name"] = cleaned["Name"].astype("string").str.strip()
    for column in ("Platform", "Genre", "Publisher"):
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned["Platform"] = cleaned["Platform"].str.upper()
    cleaned["Genre"] = cleaned["Genre"].str.title()
    cleaned["Publisher"] = cleaned["Publisher"].str.replace(r"\s+", " ", regex=True)
    missing_publishers = int(cleaned["Publisher"].isna().sum() + cleaned["Publisher"].eq("").sum())
    cleaned["Publisher"] = cleaned["Publisher"].replace("", pd.NA).fillna("Unknown")

    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce")
    cleaned["Year"] = cleaned["Year"].round().astype("Int64")
    median_year = cleaned["Year"].median()
    missing_years = int(cleaned["Year"].isna().sum())
    if pd.isna(median_year):
        raise ValueError("Year contains no valid values; unable to impute missing years")
    cleaned["Year"] = cleaned["Year"].fillna(round(float(median_year))).astype("int64")

    for column in SALES_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0)

    missing_required = cleaned[["Name", "Platform", "Genre"]].isna().any(axis=1)
    empty_required = cleaned[["Name", "Platform", "Genre"]].eq("").any(axis=1)
    dropped_required = int((missing_required | empty_required).sum())
    cleaned = cleaned.loc[~(missing_required | empty_required)].copy()
    duplicates_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    audit = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "missing_years_imputed": missing_years,
        "missing_publishers_filled": missing_publishers,
        "rows_missing_required_labels_dropped": dropped_required,
        "duplicates_removed": duplicates_removed,
    }
    return cleaned, audit
