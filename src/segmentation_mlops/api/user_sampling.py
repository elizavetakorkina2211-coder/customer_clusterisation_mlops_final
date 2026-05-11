"""Эмпирические диапазоны для случайного заполнения формы «Пользователь» (как в обучающей выборке)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# «Нормальная» зона — центральная масса распределения (без хвостов выборки)
_Q_LO = 0.10
_Q_HI = 0.90

_NUMERIC_RAW = [
    "days_since_last_order",
    "sessions_last_month",
    "avg_basket_size",
    "category_diversity",
    "discount_share",
    "returns_rate",
    "avg_session_minutes",
]
_CATEGORICAL = ["device_type", "platform", "marketing_channel", "region"]


def _csv_has_event_ts(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        return bool(head) and "event_ts" in head[0]
    except OSError:
        return False


def _fallback_config() -> dict:
    """Если нет CSV — узкие разумные границы (не полный перебор пространства)."""
    return {
        "source": "fallback",
        "numeric": {
            "days_since_last_order": {"lo": 15.0, "hi": 90.0},
            "sessions_last_month": {"lo": 2.0, "hi": 12.0},
            "avg_basket_size": {"lo": 1.5, "hi": 8.0},
            "category_diversity": {"lo": 2.0, "hi": 9.0},
            "discount_share": {"lo": 0.05, "hi": 0.35},
            "returns_rate": {"lo": 0.02, "hi": 0.22},
            "avg_session_minutes": {"lo": 8.0, "hi": 45.0},
        },
        "categorical": {
            "device_type": [
                {"v": "mobile", "p": 0.34},
                {"v": "desktop", "p": 0.33},
                {"v": "tablet", "p": 0.33},
            ],
            "platform": [
                {"v": "ios", "p": 0.34},
                {"v": "android", "p": 0.33},
                {"v": "web", "p": 0.33},
            ],
            "marketing_channel": [
                {"v": "organic", "p": 0.25},
                {"v": "paid_search", "p": 0.25},
                {"v": "social", "p": 0.25},
                {"v": "email", "p": 0.25},
            ],
            "region": [
                {"v": "EU", "p": 0.25},
                {"v": "US", "p": 0.25},
                {"v": "APAC", "p": 0.25},
                {"v": "Unknown", "p": 0.25},
            ],
        },
        "event_ts": None,
        "joint_profiles": [],
    }


def _profile_dict_from_series(r: pd.Series, df: pd.DataFrame) -> dict:
    """Один профиль для API/формы: только поля модели, без segment_truth и лишних колонок."""
    d: dict = {}
    for c in _NUMERIC_RAW:
        v = r[c]
        if pd.notna(v):
            d[c] = round(float(v), 5)
    for c in _CATEGORICAL:
        v = r[c]
        if pd.isna(v):
            d[c] = "Unknown"
        else:
            s = str(v)
            d[c] = "Unknown" if s.lower() == "nan" else s
    if "event_ts" in df.columns:
        raw_ts = r["event_ts"]
        if pd.notna(raw_ts):
            ts = pd.Timestamp(raw_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            d["event_ts"] = ts.isoformat()
    return d


def sample_random_profile_from_csv(root: Path) -> dict | None:
    """Равновероятная случайная строка из полного `customers_raw.csv` (признаки вместе, как в данных)."""
    csv_path = root / "data" / "raw" / "customers_raw.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(
        csv_path,
        parse_dates=["event_ts"] if _csv_has_event_ts(csv_path) else None,
    )
    if len(df) < 1:
        return None
    if not all(c in df.columns for c in _NUMERIC_RAW + _CATEGORICAL):
        return None
    r = df.sample(n=1).iloc[0]
    return _profile_dict_from_series(r, df)


def build_user_sampling_config(root: Path) -> dict:
    """
    Считает по `data/raw/customers_raw.csv` квантили 10–90% по числовым полям
    и частоты категорий для случайного выбора с теми же пропорциями, что в данных.
    """
    csv_path = root / "data" / "raw" / "customers_raw.csv"
    if not csv_path.is_file():
        return _fallback_config()

    df = pd.read_csv(
        csv_path,
        parse_dates=["event_ts"] if _csv_has_event_ts(csv_path) else None,
    )
    if len(df) < 20:
        return _fallback_config()

    out: dict = {"source": str(csv_path.name), "numeric": {}, "categorical": {}, "event_ts": None}

    for col in _NUMERIC_RAW:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            continue
        lo = float(s.quantile(_Q_LO))
        hi = float(s.quantile(_Q_HI))
        if col in ("discount_share", "returns_rate"):
            lo = max(0.0, lo)
            hi = min(1.0, max(lo + 1e-4, hi))
        else:
            if hi <= lo:
                hi = lo + 1e-3
        out["numeric"][col] = {"lo": lo, "hi": hi}

    for col in _CATEGORICAL:
        if col not in df.columns:
            continue
        vc = df[col].fillna("Unknown").astype(str).value_counts(normalize=True)
        opts = [{"v": str(k), "p": float(v)} for k, v in vc.items() if v > 0]
        if opts:
            s = sum(o["p"] for o in opts)
            if s > 0 and abs(s - 1.0) > 1e-6:
                for o in opts:
                    o["p"] /= s
            out["categorical"][col] = opts

    if "event_ts" in df.columns:
        ts = pd.to_datetime(df["event_ts"], utc=True, errors="coerce").dropna()
        if len(ts) >= 10:
            t_lo = ts.quantile(_Q_LO)
            t_hi = ts.quantile(_Q_HI)
            if pd.Timestamp(t_hi) > pd.Timestamp(t_lo):
                out["event_ts"] = {
                    "iso_min": pd.Timestamp(t_lo).isoformat(),
                    "iso_max": pd.Timestamp(t_hi).isoformat(),
                }

    if len(out["numeric"]) < 4:
        return _fallback_config()

    for col in _CATEGORICAL:
        if col not in out["categorical"]:
            fb = _fallback_config()
            out["categorical"][col] = fb["categorical"][col]

    out["joint_profiles"] = []

    return out
