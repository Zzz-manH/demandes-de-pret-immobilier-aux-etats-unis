for name, raw, features in [("train", train_raw, X_train), ("test", test_raw, X_test)]:
    assert features.shape == (len(raw), len(feature_columns))
    assert features.columns.tolist() == feature_columns and features.index.equals(raw.index)
    assert "id" not in features and "accepted" not in features
    assert features[numeric_columns].dtypes.eq("float64").all()
    for column in numeric_columns:
        expected = pd.to_numeric(raw[column].str.strip().replace("", np.nan), errors="raise").astype("float64")
        pd.testing.assert_series_equal(features[column], expected)
    for column in categorical_columns:
        assert features[column].map(lambda v: isinstance(v, str) and bool(v)).all()
        assert features[column].eq(MISSING_CATEGORY).equals(raw[column].str.strip().eq(""))
assert X_train.index.equals(y.index)
pd.testing.assert_series_equal(train_labels["id"], train_raw["id"])
pd.testing.assert_series_equal(test_ids["id"], test_raw["id"])

unseen_rows = []
for column in categorical_columns:
    unseen_values = sorted(set(X_test[column]) - set(X_train[column]))
    if unseen_values:
        unseen_rows.append({"Champ": column, "Nouvelles_modalités_test": len(unseen_values),
                            "Lignes_test_concernées": int(X_test[column].isin(unseen_values).sum()),
                            "Exemples": ", ".join(unseen_values[:5])})
unseen_categories = pd.DataFrame(unseen_rows, columns=["Champ", "Nouvelles_modalités_test", "Lignes_test_concernées", "Exemples"])
display_named(unseen_categories)
print("Types, valeurs manquantes, fidélité numérique et alignement des lignes vérifiés.")
