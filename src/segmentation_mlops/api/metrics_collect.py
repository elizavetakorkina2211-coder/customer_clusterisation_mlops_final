"""Снимок метрик MLOps из Prometheus client registry (для UI-дашборда)."""

from __future__ import annotations

from datetime import datetime, timezone

from prometheus_client import REGISTRY


def _parse_le(le: str) -> float:
    if le == "+Inf":
        return float("inf")
    try:
        return float(le)
    except ValueError:
        return float("nan")


def collect_mlops_snapshot() -> dict:
    """Собирает только метрики с префиксом mlops_."""
    counters: dict[str, dict[str, float]] = {}
    gauges: dict[str, float] = {}
    histograms: dict[str, dict] = {}

    for metric in REGISTRY.collect():
        if not metric.name.startswith("mlops_"):
            continue

        mtype = metric.type
        if hasattr(mtype, "value"):
            mtype = mtype.value

        if mtype == "counter":
            fam: dict[str, float] = {}
            for s in metric.samples:
                if s.name.endswith("_created"):
                    continue
                key = s.labels.get("outcome") or s.labels.get("le") or "total"
                if len(s.labels) > 1:
                    key = str(dict(sorted(s.labels.items())))
                fam[str(key)] = float(s.value)
            counters[metric.name] = fam
            # Имя семейства в registry без суффикса _total (OpenMetrics), а в Prometheus/дашбордах — с _total
            if not metric.name.endswith("_total"):
                counters[metric.name + "_total"] = fam

        elif mtype == "gauge":
            for s in metric.samples:
                if s.name.endswith("_created"):
                    continue
                gauges[metric.name] = float(s.value)

        elif mtype == "histogram":
            h = histograms.setdefault(
                metric.name,
                {"buckets": [], "sum": 0.0, "count": 0.0},
            )
            for s in metric.samples:
                if s.name.endswith("_bucket"):
                    le = s.labels.get("le", "")
                    h["buckets"].append({"le": le, "cumulative": float(s.value)})
                elif s.name.endswith("_sum"):
                    h["sum"] = float(s.value)
                elif s.name.endswith("_count"):
                    h["count"] = float(s.value)
            h["buckets"].sort(key=lambda b: _parse_le(b["le"]))

    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counters": counters,
        "gauges": gauges,
        "histograms": histograms,
    }
