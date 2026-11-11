numeric_columns = [
    "loan_amount", "loan_to_value_ratio", "property_value", "income", "loan_term",
    "total_units", "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]
categorical_columns = [c for c in feature_columns if c not in numeric_columns]
assert len(feature_columns) == 56, "Schéma d’entrée modifié ; vérifier les champs ajoutés ou supprimés"
assert len(numeric_columns) == 13 and len(categorical_columns) == 43
assert set(numeric_columns).issubset(feature_columns)
assert set(numeric_columns).isdisjoint(categorical_columns)

schema = {
    "schema_version": SCHEMA_VERSION,
    "target_column": "accepted", "id_column": "id",
    "feature_columns": feature_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "numeric_dtype": "float64", "categorical_dtype": "object",
    "missing_category": MISSING_CATEGORY,
    "csv_read_options": {"dtype": "str", "keep_default_na": False},
    "geographic_columns": ["county_code", "derived_msa-md"],
    "row_alignment": "CSV row order; train_labels.id and test_ids.id are the authoritative IDs",
    "folds": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE, "fold_values": [1, 2, 3, 4, 5]},
}
