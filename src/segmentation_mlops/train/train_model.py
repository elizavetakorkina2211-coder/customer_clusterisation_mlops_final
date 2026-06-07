"""Train pipeline, log to MLflow, save model and metrics for DVC."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from segmentation_mlops.constants import EXCLUDE_FROM_MODEL
from segmentation_mlops.features.build_features import build_features
from segmentation_mlops.metrics import calculate_metrics, confusion_matrix_payload

ROOT = Path(os.getenv("MLOPS_ROOT") or Path(__file__).resolve().parents[3])
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def _csv_has_event_ts(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        return bool(head) and "event_ts" in head[0]
    except OSError:
        return False


def load_params() -> dict:
    p = ROOT / "params.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _write_class_balance(y_train: pd.Series, y_test: pd.Series) -> None:
    def pack(y: pd.Series) -> dict:
        vc = y.value_counts()
        n = len(y)
        return {
            "counts": {str(k): int(v) for k, v in vc.items()},
            "proportion": {str(k): float(v / n) for k, v in vc.items()},
            "n": int(n),
        }

    out = {"train": pack(y_train), "test": pack(y_test)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "class_balance.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_model(df: pd.DataFrame, params: dict | None = None):
    params = params or {}
    train_p = params.get("train", {})
    test_size = float(train_p.get("test_size", 0.2))
    random_state = int(train_p.get("random_state", 22))
    max_iter = int(train_p.get("max_iter", 400))
    max_depth = int(train_p.get("max_depth", 14))
    learning_rate = float(train_p.get("learning_rate", 0.05))
    l2_regularization = float(train_p.get("l2_regularization", 0.2))
    min_samples_leaf = int(train_p.get("min_samples_leaf", 15))

    df = build_features(df)
    y = df["segment_truth"]
    drop_cols = [c for c in EXCLUDE_FROM_MODEL if c in df.columns]
    X = df.drop(columns=drop_cols)

    numeric_features = [
        "days_since_last_order",
        "sessions_last_month",
        "avg_basket_size",
        "category_diversity",
        "discount_share",
        "returns_rate",
        "avg_session_minutes",
        "order_freq",
        "discount_per_category",
        "basket_per_session",
        "recency_sessions",
        "basket_x_diversity",
        "engagement_minutes",
        "inverse_recency",
        "value_intensity",
    ]
    categorical_features = ["device_type", "platform", "marketing_channel", "region"]

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("cat", Pipeline([("onehot", onehot)]), categorical_features),
        ]
    )

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    _write_class_balance(y_train, y_test)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    classes = list(pipeline.named_steps["classifier"].classes_)
    metrics["confusion_matrix"] = confusion_matrix_payload(y_test, y_pred, classes)
    return pipeline, metrics


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    params = load_params()
    data_path = ROOT / "data" / "raw" / "customers_raw.csv"
    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}; run: dvc repro prepare or "
            "python -m segmentation_mlops.data.make_dataset"
        )

    df = pd.read_csv(data_path, parse_dates=["event_ts"] if _csv_has_event_ts(data_path) else None)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:" + str(ROOT / "mlruns")))
    mlflow.set_experiment("customer_segmentation")

    with mlflow.start_run(run_name="dvc_train") as run:
        mlflow.log_params(params.get("train", {}))
        pipeline, metrics = train_model(df, params)
        for k, v in metrics.items():
            if k == "confusion_matrix" or isinstance(v, dict):
                continue
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        try:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "CustomerSegmentation")
        except Exception as e:
            print(f"Model registry skipped: {e}")

        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODELS / "model.joblib")
        mlflow.log_artifact(str(MODELS / "model.joblib"))

        metrics_path = REPORTS / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"Run ID: {run.info.run_id}")
        print(f"Дисбаланс классов: {REPORTS / 'class_balance.json'}")


if __name__ == "__main__":
    main()
