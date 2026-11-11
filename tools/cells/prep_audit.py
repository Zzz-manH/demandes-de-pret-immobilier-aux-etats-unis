audit_rows = []
def audit(dataset, check, count, action="\u4fdd\u7559\u5e76\u8bb0\u5f55"):
    audit_rows.append({"dataset": dataset, "check": check, "count": int(count), "action": action})

for name, raw, features in [("train", train_raw, X_train), ("test", test_raw, X_test)]:
    audit(name, "duplicate_ids", raw["id"].duplicated().sum(), "\u5fc5\u987b\u4e3a0\uff1b\u5df2\u6821\u9a8c")
    audit(name, "duplicate_feature_rows_after_first", features.duplicated().sum())
    audit(name, "income_negative", features["income"].lt(0).sum())
    audit(name, "income_zero", features["income"].eq(0).sum())
    audit(name, "ltv_above_200_percent", features["loan_to_value_ratio"].gt(200).sum())
    audit(name, "ltv_above_1000_percent", features["loan_to_value_ratio"].gt(1000).sum(), "\u4fdd\u7559\uff1b\u539f\u59cb\u6765\u6e90\u5f85\u6838\u5b9e")
    audit(name, "negative_housing_age", features["tract_median_age_of_housing_units"].lt(0).sum(), "\u4fdd\u7559\uff1b\u7279\u6b8a\u7f16\u7801\u5f85\u6838\u5b9e")
    for column in ["loan_amount", "property_value", "loan_term"]:
        audit(name, f"{column}_nonpositive", features[column].le(0).sum(), "\u82e5\u975e0\uff0c\u4eba\u5de5\u6838\u67e5")
    audit(name, "minority_percent_outside_0_100", (features["tract_minority_population_percent"].lt(0) | features["tract_minority_population_percent"].gt(100)).sum(), "\u82e5\u975e0\uff0c\u4eba\u5de5\u6838\u67e5")
    for column in feature_columns:
        audit(name, f"blank::{column}", raw[column].str.strip().eq("").sum(), "\u6309\u5b57\u6bb5\u7c7b\u578b\u89c4\u8303\u7f3a\u5931")

# Hash pour l’audit des doublons, sans suppression ni association de cibles
train_feature_hash = pd.util.hash_pandas_object(X_train, index=False)
test_feature_hash = pd.util.hash_pandas_object(X_test, index=False)
audit("train_test", "shared_feature_hash_values", len(set(train_feature_hash) & set(test_feature_hash)), "\u4fdd\u7559\uff1b\u4ec5\u8bca\u65ad\uff0c\u5b58\u5728\u7406\u8bba\u54c8\u5e0c\u78b0\u649e\u53ef\u80fd")
for row in unseen_rows:
    audit("test", f"unseen_category_rows::{row['Champ']}", row["Lignes_test_concernées"], "\u4fdd\u7559\u539f\u7c7b\u522b\uff0c\u6a21\u578b\u5904\u7406\u672a\u89c1\u7c7b\u522b")
audit_table = pd.DataFrame(audit_rows)
# Conserver les valeurs persistées de audit.csv ; traduire uniquement la vue.
audit_action_labels = {
    "\u4fdd\u7559\u5e76\u8bb0\u5f55": 'Conserver et consigner',
    "\u5fc5\u987b\u4e3a0\uff1b\u5df2\u6821\u9a8c": 'Doit valoir 0 ; vérifié',
    "\u4fdd\u7559\uff1b\u539f\u59cb\u6765\u6e90\u5f85\u6838\u5b9e": 'Conserver ; source à vérifier',
    "\u4fdd\u7559\uff1b\u7279\u6b8a\u7f16\u7801\u5f85\u6838\u5b9e": 'Conserver ; code spécial à vérifier',
    "\u82e5\u975e0\uff0c\u4eba\u5de5\u6838\u67e5": 'Vérifier manuellement si non nul',
    "\u6309\u5b57\u6bb5\u7c7b\u578b\u89c4\u8303\u7f3a\u5931": 'Normaliser les valeurs manquantes selon le type du champ',
    "\u4fdd\u7559\uff1b\u4ec5\u8bca\u65ad\uff0c\u5b58\u5728\u7406\u8bba\u54c8\u5e0c\u78b0\u649e\u53ef\u80fd": 'Conserver ; diagnostic seulement, collisions de hash théoriquement possibles',
    "\u4fdd\u7559\u539f\u7c7b\u522b\uff0c\u6a21\u578b\u5904\u7406\u672a\u89c1\u7c7b\u522b": 'Conserver la catégorie ; le modèle gère les modalités inédites',
}
audit_display = audit_table[~audit_table["check"].str.startswith("blank::")].copy()
audit_display["action"] = audit_display["action"].replace(audit_action_labels)
display_named(audit_display)
