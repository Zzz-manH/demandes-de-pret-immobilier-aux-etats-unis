def read_prepared_features(path, saved_schema):
    frame = read_text_csv(path)
    assert frame.columns.tolist() == saved_schema["feature_columns"]
    for column in saved_schema["numeric_columns"]:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise").astype("float64")
    for column in saved_schema["categorical_columns"]:
        assert frame[column].ne("").all(), f"{column}: catégorie vide dans le fichier préparé"
        frame[column] = frame[column].astype(object)
    return frame

saved_schema = json.loads((PREPARED_DIR / "schema.json").read_text(encoding="utf-8"))
for name, expected in [("train_features.csv", X_train), ("test_features.csv", X_test)]:
    restored = read_prepared_features(PREPARED_DIR / name, saved_schema)
    pd.testing.assert_frame_equal(restored, expected, check_exact=False, rtol=1e-14, atol=0)
pd.testing.assert_frame_equal(read_text_csv(PREPARED_DIR / "train_labels.csv"), train_labels.astype(str))
pd.testing.assert_frame_equal(read_text_csv(PREPARED_DIR / "test_ids.csv"), test_ids)
pd.testing.assert_frame_equal(read_text_csv(PREPARED_DIR / "folds.csv"), folds.astype(str))
for name, entry in manifest["output_files"].items():
    assert sha256_file(PREPARED_DIR / name) == entry["sha256"], f"Empreinte du fichier de sortie modifiée : {name}"
print("Relecture, types, alignement des lignes et empreintes SHA-256 vérifiés.")
