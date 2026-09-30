"""Leakage-aware baseline classifiers and evaluation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, MODEL_FEATURES, prepare_model_features


def _make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("year", numeric_transformer, ["Year"]),
        ],
        remainder="drop",
    )


def build_model_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    """Create balanced logistic-regression and random-forest pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", _make_preprocessor(scale_numeric=True)),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", _make_preprocessor(scale_numeric=False)),
                ("classifier", RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]
        ),
    }


def evaluate_models(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit baselines and return holdout metrics, confusion matrices, and CV scores."""
    features, target = prepare_model_features(frame)
    if features.empty:
        raise ValueError("No rows available for modeling")
    if target.nunique() < 2:
        raise ValueError("Sales_Tier must contain at least two classes")
    class_counts = target.value_counts()
    if class_counts.min() < cv_splits:
        raise ValueError(
            f"Each class needs at least {cv_splits} rows for stratified {cv_splits}-fold CV; "
            f"smallest class has {class_counts.min()}"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    classes = sorted(target.unique())
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    models = build_model_pipelines(random_state)
    fitted_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}
    matrices: dict[str, list[list[int]]] = {}
    cv_scores: dict[str, dict[str, float]] = {}

    for name, pipeline in models.items():
        cross_validation = cross_validate(
            pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted_models[name] = pipeline
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        }
        matrices[name] = confusion_matrix(y_test, predictions, labels=classes).tolist()
        cv_scores[name] = {
            score: float(cross_validation[f"test_{score}"].mean()) for score in scoring
        }

    return {
        "models": fitted_models,
        "metrics": pd.DataFrame.from_dict(metrics, orient="index"),
        "cv_scores": pd.DataFrame.from_dict(cv_scores, orient="index"),
        "confusion_matrices": matrices,
        "classes": classes,
        "y_test": y_test,
        "features": MODEL_FEATURES.copy(),
    }
