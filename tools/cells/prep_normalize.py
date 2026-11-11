def prepare_features(raw_features):
    assert raw_features.columns.tolist() == feature_columns, "Variables ou ordre des colonnes modifiés"
    prepared = raw_features.copy()
    for column in numeric_columns:
        values = prepared[column].str.strip().replace("", np.nan)
        prepared[column] = pd.to_numeric(values, errors="raise").astype("float64")
        assert not np.isinf(prepared[column].to_numpy()).any(), f"{column}: valeurs infinies"
        assert prepared[column].isna().equals(raw_features[column].str.strip().eq("")), f"{column}: valeurs manquantes inattendues"
    for column in categorical_columns:
        values = prepared[column].str.strip()
        assert not values.eq(MISSING_CATEGORY).any(), f"{column}: conflit entre catégorie initiale et marqueur de valeur manquante"
        values = values.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        if column in schema["geographic_columns"]:
            nonblank = values.ne("")
            assert values[nonblank].str.fullmatch(r"\d{1,5}").all(), f"{column}: format de code géographique invalide"
            values = values.where(~nonblank, values.str.zfill(5))
        prepared[column] = values.replace("", MISSING_CATEGORY).astype(object)
    return prepared

X_train = prepare_features(train_raw[feature_columns])
X_test = prepare_features(test_raw[feature_columns])
print("Types harmonisés ; unités initiales conservées, sans imputation, mise à l’échelle ni encodage.")
