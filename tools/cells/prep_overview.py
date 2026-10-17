missing_table = pd.DataFrame({
    "字段": feature_columns,
    "训练空白数": [int(train_raw[c].str.strip().eq("").sum()) for c in feature_columns],
    "测试空白数": [int(test_raw[c].str.strip().eq("").sum()) for c in feature_columns],
})
missing_table["训练空白比例"] = missing_table["训练空白数"] / len(X_train)
missing_table["测试空白比例"] = missing_table["测试空白数"] / len(X_test)
display(missing_table.query("训练空白数 > 0 or 测试空白数 > 0").sort_values("训练空白比例", ascending=False))
display(X_train[numeric_columns].describe(percentiles=[0.5, 0.99, 0.999]).T.round(3))
category_profile = pd.DataFrame({
    "字段": categorical_columns,
    "训练类别数": [X_train[c].nunique() for c in categorical_columns],
    "最大类别比例": [X_train[c].value_counts(normalize=True).iloc[0] for c in categorical_columns],
}).sort_values("训练类别数", ascending=False)
display(category_profile.head(10))
