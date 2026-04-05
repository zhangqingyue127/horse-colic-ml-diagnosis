from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform


FEATURE_NAMES = [
    "surgery",
    "age",
    "rectal_temperature",
    "pulse",
    "respiratory_rate",
    "extremities_temperature",
    "peripheral_pulse",
    "mucous_membrane",
    "capillary_refill_time",
    "pain",
    "peristalsis",
    "abdominal_distension",
    "nasogastric_tube",
    "nasogastric_reflux",
    "nasogastric_reflux_ph",
    "rectal_exam_feces",
    "abdomen",
    "packed_cell_volume",
    "total_protein",
    "abdominocentesis_appearance",
    "abdominocentesis_total_protein",
    "outcome",
]


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, int]:
    train_data = pd.read_csv(train_path, header=None, sep=r"\s+", engine="python").dropna(how="all").iloc[:, :22]
    test_data = pd.read_csv(test_path, header=None, sep=r"\s+", engine="python").dropna(how="all").iloc[:, :22]
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    all_data.columns = FEATURE_NAMES
    return all_data, len(train_data)


def prepare_features(all_data: pd.DataFrame, train_size: int, variance_threshold: float = 0.85):
    x = all_data.drop("outcome", axis=1).copy()
    y = all_data["outcome"].copy()

    potential_categorical = [col for col in x.columns if x[col].nunique() < 10]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA()
    pca.fit(x_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    important_features = set()
    for i in range(n_components):
        component_weights = pca.components_[i]
        top_indices = np.argsort(np.abs(component_weights))[-3:]
        important_features.update(x.columns[top_indices])

    important_features = list(important_features)
    categorical_cols = [col for col in important_features if col in potential_categorical]

    min_categorical = 8
    if len(categorical_cols) < min_categorical:
        additional = [col for col in important_features if col not in categorical_cols][: min_categorical - len(categorical_cols)]
        categorical_cols += additional

    label_encoder = LabelEncoder()
    for col in categorical_cols:
        x[col] = label_encoder.fit_transform(x[col].astype(str))

    x_train = x.iloc[:train_size][important_features]
    x_test = x.iloc[train_size:][important_features]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    metadata = {
        "n_components": n_components,
        "variance_threshold": variance_threshold,
        "important_features": important_features,
        "categorical_cols": categorical_cols,
        "potential_categorical_cols": potential_categorical,
        "pca_explained_variance_ratio_first_5": pca.explained_variance_ratio_[:5].tolist(),
    }
    return x_train, x_test, y_train, y_test, metadata


def evaluate_model(model, x_train, x_test, y_train, y_test) -> Dict[str, object]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    else:
        y_prob = None
        auc = None

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1.0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1.0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1.0)),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }


def baseline_models() -> Dict[str, object]:
    return {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }


def tune_random_forest(x_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 3, 5, 8],
        "min_samples_split": [2, 5],
        "class_weight": [None, "balanced"],
    }
    model = RandomForestClassifier(random_state=42)
    search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(x_train, y_train)
    return search


def tune_adaboost(x_train, y_train):
    pipeline = Pipeline([
        ("ada", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            random_state=42,
        ))
    ])
    param_dist = {
        "ada__n_estimators": randint(50, 500),
        "ada__learning_rate": uniform(0.01, 0.5),
        "ada__estimator__max_depth": randint(1, 5),
        "ada__estimator__min_samples_split": randint(2, 20),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=50,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(x_train, y_train)
    return search


def tune_logistic_regression(x_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=42)),
    ])
    param_grid = {
        "lr__C": np.logspace(-3, 3, 7),
        "lr__solver": ["liblinear", "lbfgs", "saga"],
        "lr__class_weight": [None, "balanced", {-1: 2, 1: 1}],
        "lr__penalty": ["l1", "l2"],
        "lr__max_iter": [500, 1000],
    }
    valid_param_grid = []
    for solver in param_grid["lr__solver"]:
        for penalty in param_grid["lr__penalty"]:
            if penalty == "l1" and solver not in {"liblinear", "saga"}:
                continue
            valid_param_grid.append({
                "lr__C": param_grid["lr__C"],
                "lr__solver": [solver],
                "lr__class_weight": param_grid["lr__class_weight"],
                "lr__penalty": [penalty],
                "lr__max_iter": param_grid["lr__max_iter"],
            })
    search = GridSearchCV(
        pipeline,
        valid_param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def main():
    parser = argparse.ArgumentParser(description="Horse colic diagnosis ML pipeline")
    parser.add_argument("--train", required=True, help="Path to horseColicTraining2.txt")
    parser.add_argument("--test", required=True, help="Path to horseColicTest2.txt")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data, train_size = load_data(args.train, args.test)
    x_train, x_test, y_train, y_test, metadata = prepare_features(all_data, train_size)

    baseline_results = {}
    for name, model in baseline_models().items():
        baseline_results[name] = evaluate_model(model, x_train, x_test, y_train, y_test)

    tuning_results = {}

    rf_search = tune_random_forest(x_train, y_train)
    tuning_results["RandomForest"] = {
        "best_params": rf_search.best_params_,
        "best_cv_auc": float(rf_search.best_score_),
    }

    ada_search = tune_adaboost(x_train, y_train)
    tuning_results["AdaBoost"] = {
        "best_params": ada_search.best_params_,
        "best_cv_auc": float(ada_search.best_score_),
    }

    lr_search = tune_logistic_regression(x_train, y_train)
    tuning_results["LogisticRegression"] = {
        "best_params": lr_search.best_params_,
        "best_cv_auc": float(lr_search.best_score_),
    }

    summary = {
        "metadata": metadata,
        "baseline_results": baseline_results,
        "tuning_results": tuning_results,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_rows = []
    for model_name, metrics in baseline_results.items():
        metrics_rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        })
    pd.DataFrame(metrics_rows).to_csv(output_dir / "baseline_metrics.csv", index=False)

    print("Saved:")
    print(f"- {output_dir / 'summary.json'}")
    print(f"- {output_dir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
