"""Data, concept, and target drift metrics + HTML reports."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from segmentation_mlops.constants import EVENT_TS_COL
from segmentation_mlops.drift.labels import feature_ru
from segmentation_mlops.features.build_features import build_features

ROOT = Path(os.getenv("MLOPS_ROOT") or Path(__file__).resolve().parents[3])
DRIFT_REPORTS = ROOT / "reports" / "drift"


def _load_reference_train_frame() -> pd.DataFrame | None:
    """Реальная обучающая выборка после feature engineering — эталон для сравнения, не N(μ,σ)."""
    ref_path = ROOT / "data" / "raw" / "customers_raw.csv"
    if not ref_path.is_file():
        return None
    try:
        head = ref_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        has_ts = bool(head) and EVENT_TS_COL in head[0]
    except OSError:
        has_ts = False
    kwargs: dict = {}
    if has_ts:
        kwargs["parse_dates"] = [EVENT_TS_COL]
    try:
        return build_features(pd.read_csv(ref_path, **kwargs))
    except Exception:
        return None


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    expected = expected.astype(float)
    actual = actual.astype(float)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)
    e_perc = np.clip(e_counts / max(len(expected), 1), 1e-6, 1.0)
    a_perc = np.clip(a_counts / max(len(actual), 1), 1e-6, 1.0)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def load_reference() -> dict:
    p = ROOT / "data" / "processed" / "reference_profile.json"
    if not p.exists():
        return {"numeric": {}, "categorical": {}, "target_dist": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def analyze_data_drift(
    ref: dict,
    current_df: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.01,
) -> dict:
    """
    Дрейф числовых признаков: текущий батч vs подвыборка **реального** train CSV (после build_features).
    Сравнение с синтетическим N(μ,σ) из профиля давало ложные срабатывания почти по всем колонкам.
    """
    flags = []
    details = {}
    numeric = ref.get("numeric", {})
    ref_train = _load_reference_train_frame()
    rng = np.random.default_rng(42)

    for col, stat in numeric.items():
        if col not in current_df.columns:
            continue
        cur = current_df[col].astype(float).dropna().values
        if len(cur) < 10:
            continue

        if ref_train is not None and col in ref_train.columns:
            ref_s = ref_train[col].astype(float).dropna().to_numpy()
            if len(ref_s) < 30:
                continue
            if len(ref_s) > 3000:
                ref_s = rng.choice(ref_s, size=3000, replace=False)
        else:
            ref_s = rng.normal(stat["mean"], stat["std"], size=max(len(cur), 200)).astype(float)

        psi_v = _psi(ref_s, cur)
        ks_stat, ks_p = stats.ks_2samp(ref_s, cur)
        # Оба критерия должны указывать на сдвиг (раньше было ИЛИ — лавина ложных тревог)
        flagged = (psi_v > psi_threshold) and (ks_p < ks_alpha)
        details[col] = {"psi": psi_v, "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p)}
        if flagged:
            flags.append(f'Дрейф данных: признак «{feature_ru(col)}»')
    return {"flags": flags, "details": details, "summary": {"n_features_flagged": len(flags)}}


def analyze_concept_drift(
    ref: dict,
    predicted_labels: list[str],
    l1_threshold: float = 0.32,
) -> dict:
    """Drift in prediction distribution vs training target marginal (proxy for concept drift)."""
    ref_t = ref.get("target_dist", {})
    if not ref_t or not predicted_labels:
        return {"flags": [], "details": {}, "summary": {"note": "no_reference_or_predictions"}}
    cur = Counter(predicted_labels)
    n = sum(cur.values())
    cur_p = {k: cur[k] / n for k in cur}
    all_k = set(ref_t) | set(cur_p)
    diff = sum(abs(cur_p.get(k, 0) - ref_t.get(k, 0)) for k in all_k)
    flagged = diff > l1_threshold
    return {
        "flags": (
            ["Дрейф концепции: доли классов в предсказаниях заметно отличаются от обучения"]
            if flagged
            else []
        ),
        "details": {"l1_dist_marginals": float(diff), "current": cur_p, "reference": ref_t},
        "summary": {"flagged": flagged},
    }


def analyze_temporal_drift(
    ref: dict,
    current_df: pd.DataFrame,
    recent_fraction: float = 0.25,
    psi_threshold: float = 0.15,
    min_total_rows: int = 120,
    min_bucket_rows: int = 40,
    ks_alpha: float = 0.005,
) -> dict:
    """
    Сравнение распределений признаков между «ранними» и «поздними» записями по event_ts
    (сдвиг во времени внутри текущего батча).

    На малых батчах (десятки строк) PSI «ранние vs поздние» шумит и даёт ложные срабатывания
    почти по всем признакам; требуем достаточный размер обеих половин и согласие PSI с KS.
    """
    flags: list[str] = []
    details: dict = {}

    if EVENT_TS_COL not in current_df.columns:
        return {"flags": [], "details": {}, "summary": {"note": "no_event_ts_column"}}

    ts = pd.to_datetime(current_df[EVENT_TS_COL], utc=True, errors="coerce")
    work = current_df.assign(_ev_ts=ts).dropna(subset=["_ev_ts"])
    if len(work) < min_total_rows:
        return {
            "flags": [],
            "details": {"n": len(work)},
            "summary": {
                "note": "sample_below_temporal_min_total",
                "min_total_rows": min_total_rows,
            },
        }

    work = work.sort_values("_ev_ts")
    k = max(2, int(len(work) * recent_fraction))
    recent = work.iloc[-k:]
    older = work.iloc[:-k]
    if len(older) < 5:
        return {"flags": [], "details": {}, "summary": {"note": "insufficient_older_window"}}

    if len(recent) < min_bucket_rows or len(older) < min_bucket_rows:
        return {
            "flags": [],
            "details": {"older_n": len(older), "recent_n": len(recent)},
            "summary": {
                "note": "sample_below_temporal_min_bucket",
                "min_bucket_rows": min_bucket_rows,
            },
        }

    tmin = work["_ev_ts"].min()
    tmax = work["_ev_ts"].max()
    details["current_window_utc"] = {
        "min": tmin.isoformat(),
        "max": tmax.isoformat(),
        "n": len(work),
    }
    if ref.get("time"):
        details["reference_time_utc"] = ref["time"]

    numeric_cols = [
        c
        for c in ref.get("numeric", {})
        if c in work.columns and c != EVENT_TS_COL
    ]
    col_psi: dict[str, float] = {}
    col_ks_p: dict[str, float] = {}
    for col in numeric_cols:
        a = older[col].astype(float).dropna().values
        b = recent[col].astype(float).dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        psi_v = _psi(a, b)
        col_psi[col] = float(psi_v)
        try:
            _, ks_p = stats.ks_2samp(a, b)
        except ValueError:
            ks_p = 1.0
        col_ks_p[col] = float(ks_p)
        flagged = (psi_v > psi_threshold) and (ks_p < ks_alpha)
        if flagged:
            flags.append(
                f'Временной дрейф: признак «{feature_ru(col)}» '
                "(ранние и поздние записи по времени события различаются)"
            )

    details["psi_recent_vs_older"] = col_psi
    details["ks_pvalue_recent_vs_older"] = col_ks_p
    return {
        "flags": flags,
        "details": details,
        "summary": {"n_flags": len(flags), "recent_n": len(recent), "older_n": len(older)},
    }


def analyze_target_drift(
    ref: dict,
    y_true_series: pd.Series | None,
) -> dict:
    """Target distribution drift when labels exist."""
    ref_t = ref.get("target_dist", {})
    if y_true_series is None or y_true_series.empty or not ref_t:
        return {"flags": [], "details": {}, "summary": {"note": "labels_unavailable"}}
    vc = y_true_series.value_counts(normalize=True)
    cur_p = {str(k): float(v) for k, v in vc.items()}
    all_k = set(ref_t) | set(cur_p)
    diff = sum(abs(cur_p.get(k, 0) - ref_t.get(k, 0)) for k in all_k)
    flagged = diff > 0.2
    return {
        "flags": (
            ["Дрейф целевой переменной: распределение меток не совпадает с обучением"]
            if flagged
            else []
        ),
        "details": {"l1_dist": float(diff), "current": cur_p},
        "summary": {"flagged": flagged},
    }


def build_full_report(
    current_features_df: pd.DataFrame,
    predicted_labels: list[str],
    y_true: pd.Series | None,
    params: dict | None = None,
) -> dict:
    params = params or {}
    drift_p = params.get("drift", {})
    psi_t = float(drift_p.get("psi_threshold", 0.28))
    ks_a = float(drift_p.get("ks_alpha", 0.005))
    recent_frac = float(drift_p.get("recent_fraction", 0.25))
    time_psi_t = float(drift_p.get("time_psi_threshold", 0.38))
    time_ks_a = float(drift_p.get("time_ks_alpha", ks_a))
    temporal_min_total = int(drift_p.get("temporal_min_total_rows", 120))
    temporal_min_bucket = int(drift_p.get("temporal_min_bucket_rows", 40))
    concept_l1 = float(drift_p.get("concept_l1_threshold", 0.36))

    ref = load_reference()
    data = analyze_data_drift(ref, current_features_df, psi_threshold=psi_t, ks_alpha=ks_a)
    concept = analyze_concept_drift(ref, predicted_labels, l1_threshold=concept_l1)
    target = analyze_target_drift(ref, y_true)
    temporal = analyze_temporal_drift(
        ref,
        current_features_df,
        recent_fraction=recent_frac,
        psi_threshold=time_psi_t,
        min_total_rows=temporal_min_total,
        min_bucket_rows=temporal_min_bucket,
        ks_alpha=time_ks_a,
    )

    all_flags = data["flags"] + concept["flags"] + target["flags"] + temporal["flags"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp_utc": ts,
        "data_drift": data,
        "concept_drift": concept,
        "target_drift": target,
        "temporal_drift": temporal,
        "anomaly_flags": all_flags,
    }

    DRIFT_REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = DRIFT_REPORTS / f"drift_report_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _try_evidently_html(current_features_df, ts)

    latest = DRIFT_REPORTS / "latest_drift.json"
    latest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _try_evidently_html(current_df: pd.DataFrame, ts: str) -> None:
    ref_path = ROOT / "data" / "raw" / "customers_raw.csv"
    if not ref_path.exists() or current_df.empty:
        return
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        ref_df = build_features(pd.read_csv(ref_path))
        common = [
            c
            for c in ref_df.columns
            if c in current_df.columns and c not in ("segment_truth", EVENT_TS_COL)
        ]
        if len(common) < 3:
            return
        ref_s = ref_df[common].head(5000)
        cur_s = current_df[common].iloc[: max(1, len(current_df))]
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_s, current_data=cur_s)
        html_path = DRIFT_REPORTS / f"drift_evidently_{ts}.html"
        report.save_html(str(html_path))
    except Exception:
        pass
