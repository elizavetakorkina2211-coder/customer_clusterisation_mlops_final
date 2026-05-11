"""Построчные флаги аномалий при инференсе (эталон из reference_profile)."""

from __future__ import annotations

from typing import Any

from segmentation_mlops.drift.labels import feature_ru


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_prediction_flags(
    row: dict[str, Any],
    proba: dict[str, float],
    ref: dict,
    params: dict | None = None,
) -> list[str]:
    """
    row — признаки после build_features (одна строка), совместимые с reference_profile.
    """
    params = params or {}
    z_t = float(params.get("zscore_threshold", 3.0))
    min_top = float(params.get("min_top_proba", 0.45))
    rare_p = float(params.get("rare_category_max_p", 0.03))

    flags: list[str] = []

    for col, stat in ref.get("numeric", {}).items():
        if col not in row:
            continue
        v = _as_float(row[col])
        if v is None:
            continue
        mean = float(stat.get("mean", 0))
        std = float(stat.get("std") or 1e-9)
        if std <= 0:
            std = 1e-9
        z = abs(v - mean) / std
        if z > z_t:
            flags.append(
                f'Сильное отклонение от эталона: «{feature_ru(col)}» (~{z:.1f}σ)'
            )

    for col, dist in ref.get("categorical", {}).items():
        if col not in row:
            continue
        val = str(row[col])
        p_cat = dist.get(val)
        if p_cat is None:
            p_cat = 0.0
        if p_cat < rare_p:
            flags.append(
                f'Редкое значение на обучении: «{feature_ru(col)}» = «{val}»'
            )

    if proba:
        mx = max(proba.values())
        if mx < min_top:
            flags.append(
                f"Низкая уверенность модели (лучший класс {mx:.0%} < {min_top:.0%})"
            )

    return flags
