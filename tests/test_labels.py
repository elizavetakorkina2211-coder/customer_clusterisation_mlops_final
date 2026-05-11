from segmentation_mlops.drift.labels import categorical_value_ru, feature_title_ru


def test_feature_title_ru_capitalizes():
    t = feature_title_ru("days_since_last_order")
    assert t[0].isupper()
    assert "заказа" in t.lower()


def test_categorical_value_ru():
    assert categorical_value_ru("device_type", "mobile") == "мобильное"
    assert categorical_value_ru("region", "EU") == "ЕС"
    assert categorical_value_ru("marketing_channel", "email") == "e-mail"
