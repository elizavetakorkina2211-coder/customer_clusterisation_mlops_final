"""Преобразование флагов дрейфа для UI (в т.ч. старые английские коды в отчётах)."""

from __future__ import annotations

from segmentation_mlops.drift.labels import feature_ru

_LEGACY_EXACT: dict[str, str] = {
    "concept_drift:prediction_marginal": (
        "Дрейф концепции: доли классов в предсказаниях заметно отличаются от обучения"
    ),
    "target_drift:label_distribution": (
        "Дрейф целевой переменной: распределение меток не совпадает с обучением"
    ),
}


def humanize_drift_flag(flag: str) -> str:
    if flag in _LEGACY_EXACT:
        return _LEGACY_EXACT[flag]
    if flag.startswith("data_drift:"):
        col = flag.split(":", 1)[1]
        return f'Дрейф данных: признак «{feature_ru(col)}»'
    if flag.startswith("time_drift:"):
        col = flag.split(":", 1)[1]
        return (
            f'Временной дрейф: признак «{feature_ru(col)}» '
            "(ранние и поздние записи по времени события различаются)"
        )
    return flag


def humanize_drift_flags(flags: list[str]) -> list[str]:
    return [humanize_drift_flag(f) for f in flags]
