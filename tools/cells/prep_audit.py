audit_rows = []
def audit(dataset, check, count, action="保留并记录"):
    audit_rows.append({"dataset": dataset, "check": check, "count": int(count), "action": action})

for name, raw, features in [("train", train_raw, X_train), ("test", test_raw, X_test)]:
    audit(name, "duplicate_ids", raw["id"].duplicated().sum(), "必须为0；已校验")
    audit(name, "duplicate_feature_rows_after_first", features.duplicated().sum())
    audit(name, "income_negative", features["income"].lt(0).sum())
    audit(name, "income_zero", features["income"].eq(0).sum())
    audit(name, "ltv_above_200_percent", features["loan_to_value_ratio"].gt(200).sum())
    audit(name, "ltv_above_1000_percent", features["loan_to_value_ratio"].gt(1000).sum(), "保留；原始来源待核实")
    audit(name, "negative_housing_age", features["tract_median_age_of_housing_units"].lt(0).sum(), "保留；特殊编码待核实")
    for column in ["loan_amount", "property_value", "loan_term"]:
        audit(name, f"{column}_nonpositive", features[column].le(0).sum(), "若非0，人工核查")
    audit(name, "minority_percent_outside_0_100", (features["tract_minority_population_percent"].lt(0) | features["tract_minority_population_percent"].gt(100)).sum(), "若非0，人工核查")
    for column in feature_columns:
        audit(name, f"blank::{column}", raw[column].str.strip().eq("").sum(), "按字段类型规范缺失")

# 哈希用于紧凑的重复组合审计，不用于删除或匹配标签。
train_feature_hash = pd.util.hash_pandas_object(X_train, index=False)
test_feature_hash = pd.util.hash_pandas_object(X_test, index=False)
audit("train_test", "shared_feature_hash_values", len(set(train_feature_hash) & set(test_feature_hash)), "保留；仅诊断，存在理论哈希碰撞可能")
for row in unseen_rows:
    audit("test", f"unseen_category_rows::{row['字段']}", row["涉及测试行数"], "保留原类别，模型处理未见类别")
audit_table = pd.DataFrame(audit_rows)
display(audit_table[~audit_table["check"].str.startswith("blank::")])
