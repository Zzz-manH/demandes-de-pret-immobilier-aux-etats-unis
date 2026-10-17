def prepare_features(raw_features):
    assert raw_features.columns.tolist() == feature_columns, "特征或列顺序变化"
    prepared = raw_features.copy()
    for column in numeric_columns:
        values = prepared[column].str.strip().replace("", np.nan)
        prepared[column] = pd.to_numeric(values, errors="raise").astype("float64")
        assert not np.isinf(prepared[column].to_numpy()).any(), f"{column}: 存在无穷值"
        assert prepared[column].isna().equals(raw_features[column].str.strip().eq("")), f"{column}: 意外缺失"
    for column in categorical_columns:
        values = prepared[column].str.strip()
        assert not values.eq(MISSING_CATEGORY).any(), f"{column}: 原始类别与缺失标记冲突"
        values = values.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        if column in schema["geographic_columns"]:
            nonblank = values.ne("")
            assert values[nonblank].str.fullmatch(r"\d{1,5}").all(), f"{column}: 地理代码格式异常"
            values = values.where(~nonblank, values.str.zfill(5))
        prepared[column] = values.replace("", MISSING_CATEGORY).astype(object)
    return prepared

X_train = prepare_features(train_raw[feature_columns])
X_test = prepare_features(test_raw[feature_columns])
print("格式整理完成；原始数值仍保持原单位，未进行填补、缩放或编码。")
