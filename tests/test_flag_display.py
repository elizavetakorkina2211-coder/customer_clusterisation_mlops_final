from segmentation_mlops.drift.flag_display import humanize_drift_flag, humanize_drift_flags


def test_legacy_concept_drift():
    assert "концепции" in humanize_drift_flag("concept_drift:prediction_marginal")


def test_legacy_target_drift():
    assert "целевой" in humanize_drift_flag("target_drift:label_distribution")


def test_data_drift_prefix():
    s = humanize_drift_flag("data_drift:region")
    assert "Дрейф данных" in s
    assert "регион" in s


def test_time_drift_prefix():
    s = humanize_drift_flag("time_drift:event_ts")
    assert "Временной дрейф" in s


def test_passthrough_russian():
    ru = "Дрейф концепции: тест"
    assert humanize_drift_flag(ru) == ru


def test_humanize_list():
    out = humanize_drift_flags(
        ["concept_drift:prediction_marginal", "Другой текст"]
    )
    assert len(out) == 2
    assert "концепции" in out[0]
    assert out[1] == "Другой текст"
