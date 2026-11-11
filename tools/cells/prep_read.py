SOURCE_FILES = ["train.csv", "test.csv", "sample_submission.csv"]
for name in SOURCE_FILES:
    if not (DATA_DIR / name).is_file():
        raise FileNotFoundError(f"Fichier manquant : {DATA_DIR / name} ; placer les données source dans le répertoire data/ du projet.")

def read_text_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

train_raw = read_text_csv(DATA_DIR / "train.csv")
test_raw = read_text_csv(DATA_DIR / "test.csv")
sample_raw = read_text_csv(DATA_DIR / "sample_submission.csv")
assert {"id", "accepted"}.issubset(train_raw.columns)
assert "id" in test_raw and "accepted" not in test_raw
assert sample_raw.columns.tolist() == ["id", "accepted"], "Colonnes du modèle de soumission modifiées ; vérifier le contrat"
feature_columns = train_raw.columns.drop(["id", "accepted"]).tolist()
assert feature_columns == test_raw.columns.drop("id").tolist(), "Variables train/test ou ordre des colonnes incohérents"
assert len(train_raw) > 0 and len(test_raw) > 0
for name, frame in [("train", train_raw), ("test", test_raw), ("sample", sample_raw)]:
    assert frame.columns.is_unique, f"{name}: noms de colonnes en double"
    assert frame["id"].str.strip().ne("").all() and frame["id"].is_unique, f"{name}: ID invalide"
    assert frame["id"].eq(frame["id"].str.strip()).all(), f"{name}: espaces en début ou fin d’ID"
assert set(train_raw["id"]).isdisjoint(test_raw["id"]), "ID communs au train et au test"
assert set(sample_raw["id"]) == set(test_raw["id"]), "ID du modèle de soumission différents des ID test"
assert train_raw["accepted"].isin(["0", "1"]).all(), "accepted doit valoir 0 ou 1"
y = train_raw["accepted"].astype("int8")
assert set(y.unique()) == {0, 1}, "Le train binaire doit contenir les deux classes"
train_labels = pd.DataFrame({"id": train_raw["id"], "accepted": y})
test_ids = test_raw[["id"]].copy()
display_named(pd.DataFrame({
    "Jeu": ["train", "test"], "Lignes": [len(train_raw), len(test_raw)],
    "Variables": [len(feature_columns)] * 2,
}))
display_named(y.value_counts().sort_index().rename("Effectif").to_frame().assign(Proportion=lambda t: t["Effectif"] / len(y)))
