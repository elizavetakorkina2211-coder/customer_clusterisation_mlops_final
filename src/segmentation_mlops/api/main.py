"""FastAPI service: OpenAPI, inference, drift, retrain, Prometheus, web UI."""

from __future__ import annotations

import json
import numbers
import os
import subprocess
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from segmentation_mlops.api import metrics_prom as prom
from segmentation_mlops.api.experiments_display import format_mlflow_run_for_ui
from segmentation_mlops.api.metrics_collect import collect_mlops_snapshot
from segmentation_mlops.api.schemas import DriftRunOut, PredictIn, PredictOut, RetrainOut
from segmentation_mlops.api.user_sampling import (
    build_user_sampling_config,
    sample_random_profile_from_csv,
)
from segmentation_mlops.config import get_settings
from segmentation_mlops.drift.analyzer import build_full_report, load_reference
from segmentation_mlops.drift.flag_display import humanize_drift_flags
from segmentation_mlops.drift.labels import categorical_value_ru, feature_ru, feature_title_ru
from segmentation_mlops.drift.prediction_flags import compute_prediction_flags
from segmentation_mlops.features.build_features import build_features
from segmentation_mlops.store.db import (
    clear_all_predictions,
    init_db,
    list_predictions,
    merge_prediction_flags,
    save_prediction,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _tojson_filter(value, indent: int | None = None) -> str:
    kw: dict = {"ensure_ascii": False, "default": str}
    if indent is not None:
        kw["indent"] = int(indent)
    return json.dumps(value, **kw)


def _payload_value_ru_filter(value, key: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return categorical_value_ru(key, value)
    return str(value)


templates.env.filters["tojson"] = _tojson_filter
templates.env.filters["feature_ru"] = feature_ru
templates.env.filters["feature_title_ru"] = feature_title_ru
templates.env.filters["payload_value_ru"] = _payload_value_ru_filter

RECENT_LABELS: deque[str] = deque(maxlen=500)


def _load_params(root: Path) -> dict:
    p = root / "params.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _reload_model(app: FastAPI) -> None:
    settings = app.state.settings
    mp = settings.resolved_model_path()
    if mp.exists():
        app.state.model = joblib.load(mp)
        prom.MODEL_LOADED.set(1)
    else:
        app.state.model = None
        prom.MODEL_LOADED.set(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    init_db(settings.resolved_db_path())
    _reload_model(app)
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri())
    yield


app = FastAPI(
    title="Customer Segmentation MLOps API",
    description="Inference, drift monitoring, retraining, MLflow integration.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/sampling/random-csv-row")
def random_csv_row_sampling_api():
    """Случайная строка признаков из полного `data/raw/customers_raw.csv` (для формы «Пользователь»)."""
    prof = sample_random_profile_from_csv(app.state.settings.root)
    if prof is None:
        raise HTTPException(status_code=404, detail="customers_raw.csv not found or missing required columns")
    return JSONResponse(prof)


@app.get("/api/v1/metrics/snapshot")
def metrics_snapshot_api():
    """JSON-снимок метрик MLOps для веб-дашборда (тот же источник, что и /metrics)."""
    return collect_mlops_snapshot()


@app.get("/api/v1/reports/training-metrics")
def training_metrics_report_api():
    """Последние метрики качества из `reports/metrics.json` (после обучения / DVC)."""
    p = app.state.settings.root / "reports" / "metrics.json"
    if not p.exists():
        return JSONResponse({})
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


def _series_to_flag_dict(s: pd.Series) -> dict:
    """Одна строка после build_features — для сравнения с reference_profile."""
    out: dict = {}
    for k, v in s.items():
        if k == "segment_truth":
            continue
        if pd.isna(v):
            continue
        if isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, numbers.Integral):
            out[k] = int(v)
        elif isinstance(v, numbers.Real):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def _row_from_payload(p: PredictIn) -> pd.DataFrame:
    d = p.model_dump()
    if d.get("event_ts") is None:
        d["event_ts"] = datetime.now(timezone.utc)
    return pd.DataFrame([d])


@app.post("/api/v1/predict", response_model=PredictOut)
def predict_api(body: PredictIn):
    settings = app.state.settings
    if app.state.model is None:
        prom.PREDICTION_REQUESTS.labels(outcome="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    t0 = time.perf_counter()
    raw = _row_from_payload(body)
    feat_df = build_features(raw)
    X = feat_df.drop(columns=["segment_truth"], errors="ignore")

    try:
        pred = app.state.model.predict(X)[0]
        proba = app.state.model.predict_proba(X)[0]
        classes = list(app.state.model.named_steps["classifier"].classes_)
        proba_dict = {str(c): float(pr) for c, pr in zip(classes, proba, strict=True)}
    except Exception as e:
        prom.PREDICTION_REQUESTS.labels(outcome="error").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload = body.model_dump(mode="json")
    if payload.get("event_ts") is None:
        payload["event_ts"] = datetime.now(timezone.utc).isoformat()

    ref = load_reference()
    pr_params = _load_params(settings.root).get("prediction_row_flags", {})
    row_for_flags = _series_to_flag_dict(feat_df.iloc[0])
    anomaly_flags = compute_prediction_flags(row_for_flags, proba_dict, ref, pr_params)

    pid = save_prediction(
        payload,
        str(pred),
        proba_dict,
        anomaly_flags,
        settings.resolved_db_path(),
    )

    RECENT_LABELS.append(str(pred))
    if RECENT_LABELS:
        whales = sum(1 for x in RECENT_LABELS if x == "whales")
        prom.BUSINESS_WHALE_RATE.set(whales / len(RECENT_LABELS))

    prom.PREDICTION_LATENCY.observe(time.perf_counter() - t0)
    prom.PREDICTION_REQUESTS.labels(outcome="ok").inc()

    return PredictOut(
        prediction=str(pred),
        probabilities=proba_dict,
        anomaly_flags=anomaly_flags,
        prediction_id=pid,
    )


def _run_training_sync(root: Path) -> tuple[int, str]:
    env = {**os.environ, "MLFLOW_TRACKING_URI": get_settings().resolved_mlflow_uri()}
    proc = subprocess.run(
        [sys.executable, "-m", "segmentation_mlops.train.train_model"],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@app.post("/api/v1/retrain", response_model=RetrainOut)
def retrain_api(background: BackgroundTasks):
    settings = app.state.settings

    def job():
        code, out = _run_training_sync(settings.root)
        if code == 0:
            _reload_model(app)
        app.state.last_retrain_log = out[-4000:]

    background.add_task(job)
    return RetrainOut(
        status="started",
        detail="Обучение поставлено в очередь. Результат — в логах и на странице «Эксперименты».",
    )


@app.post("/api/v1/drift/run", response_model=DriftRunOut)
def drift_run_api(background: BackgroundTasks):
    settings = app.state.settings
    rows = list_predictions(settings.resolved_db_path(), limit=300)
    if len(rows) < 5:
        return DriftRunOut(
            status="skipped",
            detail="Мало предсказаний в БД для дрейфа. Сделайте несколько инференсов.",
        )

    payloads = [r["payload"] for r in rows]
    preds = [r["prediction"] for r in rows]
    df_raw = pd.DataFrame(payloads)
    y_true = None
    if "segment_truth" in df_raw.columns:
        y_true = df_raw["segment_truth"]

    df_feat = build_features(df_raw.drop(columns=["segment_truth"], errors="ignore"))
    params = _load_params(settings.root)
    report = build_full_report(df_feat, preds, y_true, params)
    flags = humanize_drift_flags(report.get("anomaly_flags", []))
    prom.DRIFT_FLAGS.set(len(flags))
    prediction_ids = [int(r["id"]) for r in rows]
    if flags:
        prefix = "Дрейф (батч): "
        batch_flags = [prefix + f for f in flags[:12]]
        if len(flags) > 12:
            batch_flags.append(f"{prefix}… всего {len(flags)} предупреждений")
        merge_prediction_flags(prediction_ids, batch_flags, settings.resolved_db_path())
    latest = settings.root / "reports" / "drift" / "latest_drift.json"

    drift_p = params.get("drift", {})
    min_flags = int(drift_p.get("auto_retrain_min_flags", 0))
    auto_triggered = False
    if min_flags > 0 and len(flags) >= min_flags:

        def auto_retrain():
            code, _ = _run_training_sync(settings.root)
            if code == 0:
                _reload_model(app)

        background.add_task(auto_retrain)
        auto_triggered = True

    return DriftRunOut(
        status="ok",
        report_path=str(latest),
        flags=flags,
        auto_retrain_triggered=auto_triggered,
    )


@app.get("/api/v1/drift/latest")
def drift_latest_api():
    settings = app.state.settings
    p = settings.root / "reports" / "drift" / "latest_drift.json"
    if not p.exists():
        return JSONResponse({})
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data.get("anomaly_flags"), list):
        data = {
            **data,
            "anomaly_flags": humanize_drift_flags(data["anomaly_flags"]),
        }
    return JSONResponse(data)


def _mlflow_runs(settings) -> list[dict]:
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri())
    exp = mlflow.get_experiment_by_name("customer_segmentation")
    if exp is None:
        return []
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=30)
    keep = {"run_id", "start_time", "status"}
    cols = [
        c
        for c in df.columns
        if c.startswith("metrics.") or c.startswith("params.") or c in keep
    ]
    slim = df[cols] if cols else df
    return slim.to_dict("records")


@app.get("/api/v1/experiments/runs")
def experiments_runs_api():
    return _mlflow_runs(app.state.settings)


@app.get("/", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse(request, "home.html", {})


@app.get("/ui/inference", response_class=HTMLResponse)
def ui_inference(request: Request):
    user_sampling = build_user_sampling_config(app.state.settings.root)
    return templates.TemplateResponse(
        request,
        "inference.html",
        {"user_sampling": user_sampling},
    )


@app.get("/ui/predictions", response_class=HTMLResponse)
def ui_predictions(request: Request):
    rows = list_predictions(app.state.settings.resolved_db_path(), limit=40)
    return templates.TemplateResponse(request, "predictions.html", {"rows": rows})


@app.post("/ui/predictions/clear")
def ui_predictions_clear():
    clear_all_predictions(app.state.settings.resolved_db_path())
    return RedirectResponse(url="/ui/predictions", status_code=303)


@app.get("/ui/experiments", response_class=HTMLResponse)
def ui_experiments(request: Request):
    runs_raw = _mlflow_runs(app.state.settings)
    runs = [format_mlflow_run_for_ui(r) for r in runs_raw]
    return templates.TemplateResponse(request, "experiments.html", {"runs": runs})


@app.get("/static/drift-reports/{filename}")
def drift_report_static(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="Invalid path")
    root = app.state.settings.root
    p = root / "reports" / "drift" / filename
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    mt = "text/html" if filename.endswith(".html") else "application/json"
    return FileResponse(p, media_type=mt)


@app.get("/ui/drift", response_class=HTMLResponse)
def ui_drift(request: Request):
    settings = app.state.settings
    drift_dir = settings.root / "reports" / "drift"
    latest: dict = {}
    latest_path = drift_dir / "latest_drift.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        af = latest.get("anomaly_flags")
        if isinstance(af, list):
            latest = {
                **latest,
                "anomaly_flags": humanize_drift_flags(af),
            }
    return templates.TemplateResponse(
        request,
        "drift.html",
        {"latest": latest},
    )
