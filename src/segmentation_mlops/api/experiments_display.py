"""Человекочитаемое представление строк MLflow для веб-страницы «Эксперименты»."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

STATUS_RU: dict[str, str] = {
    "FINISHED": "Завершён",
    "RUNNING": "Выполняется",
    "FAILED": "Ошибка",
    "KILLED": "Прерван",
    "SCHEDULED": "В очереди",
}

METRIC_ORDER: list[str] = [
    "metrics.accuracy",
    "metrics.f1_weighted",
    "metrics.f1_whales",
    "metrics.log_loss",
]

METRIC_RU: dict[str, str] = {
    "metrics.accuracy": "Точность",
    "metrics.f1_weighted": "F1 (взвешенный)",
    "metrics.f1_whales": "F1, класс «киты»",
    "metrics.log_loss": "Log loss",
}

PARAM_RU: dict[str, str] = {
    "params.test_size": "Доля тестовой выборки",
    "params.random_state": "Сид случайности",
    "params.n_estimators": "Число деревьев (Random Forest)",
    "params.max_depth": "Максимальная глубина деревьев",
    "params.max_iter": "Итераций градиентного бустинга",
    "params.learning_rate": "Темп обучения",
    "params.l2_regularization": "L2-регуляризация",
    "params.min_samples_leaf": "Мин. объектов в листе",
}


def _scalar(v: Any) -> Any:
    if hasattr(v, "item") and callable(getattr(v, "item", None)):
        try:
            return v.item()
        except Exception:
            return v
    return v


def _is_nan(v: Any) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except TypeError:
        return False


def _fmt_val(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        if abs(v) < 1e-6 and v != 0:
            return f"{v:.4e}".replace(".", ",")
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")
    return str(v)


def _format_start(st: Any) -> str:
    if st is None:
        return "—"
    try:
        if st is pd.NaT:
            return "—"
    except Exception:
        pass
    try:
        ts = pd.Timestamp(st)
        if pd.isna(ts):
            return "—"
        return ts.strftime("%d.%m.%Y %H:%M UTC")
    except Exception:
        return str(st)


def format_mlflow_run_for_ui(run: dict) -> dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    status = str(run.get("status") or "")

    metrics_items: list[tuple[str, str]] = []
    seen_m: set[str] = set()
    for k in METRIC_ORDER:
        if k not in run:
            continue
        v = _scalar(run[k])
        if v is None or _is_nan(v):
            continue
        seen_m.add(k)
        metrics_items.append(
            (METRIC_RU.get(k, k.replace("metrics.", "").replace("_", " ")), _fmt_val(v))
        )

    for k in sorted(x for x in run if x.startswith("metrics.")):
        if k in seen_m:
            continue
        v = _scalar(run[k])
        if v is None or _is_nan(v):
            continue
        short = k.removeprefix("metrics.")
        label = METRIC_RU.get(k, short.replace("_", " "))
        metrics_items.append((label, _fmt_val(v)))

    params_items: list[tuple[str, str]] = []
    for k in sorted(x for x in run if x.startswith("params.")):
        v = _scalar(run[k])
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        label = PARAM_RU.get(k, k.removeprefix("params.").replace("_", " "))
        params_items.append((label, _fmt_val(v)))

    rid_short = run_id[:10] + "…" if len(run_id) > 10 else run_id

    return {
        "run_id": run_id,
        "run_id_short": rid_short,
        "status": status,
        "status_ru": STATUS_RU.get(status, status or "—"),
        "started": _format_start(run.get("start_time")),
        "metrics_items": metrics_items,
        "params_items": params_items,
    }
