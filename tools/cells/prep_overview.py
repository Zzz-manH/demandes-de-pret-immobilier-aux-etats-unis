missing_table = pd.DataFrame({
    "Champ": feature_columns,
    "Vides_train": [int(train_raw[c].str.strip().eq("").sum()) for c in feature_columns],
    "Vides_test": [int(test_raw[c].str.strip().eq("").sum()) for c in feature_columns],
})
missing_table["Taux_vides_train"] = missing_table["Vides_train"] / len(X_train)
missing_table["Taux_vides_test"] = missing_table["Vides_test"] / len(X_test)
display_named(missing_table.query("Vides_train > 0 or Vides_test > 0").sort_values("Taux_vides_train", ascending=False))
display_named(X_train[numeric_columns].describe(percentiles=[0.5, 0.99, 0.999]).T.round(3))
category_profile = pd.DataFrame({
    "Champ": categorical_columns,
    "Modalités_train": [X_train[c].nunique() for c in categorical_columns],
    "Part_modalité_majoritaire": [X_train[c].value_counts(normalize=True).iloc[0] for c in categorical_columns],
}).sort_values("Modalités_train", ascending=False)
display_named(category_profile.head(10))
