assert y.value_counts().min() >= N_SPLITS
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_assignment = np.zeros(len(y), dtype="int8")
validation_seen = np.zeros(len(y), dtype="int8")
for fold, (fit_idx, valid_idx) in enumerate(cv.split(X_train, y), start=1):
    assert len(np.intersect1d(fit_idx, valid_idx)) == 0
    assert len(fit_idx) + len(valid_idx) == len(y)
    fold_assignment[valid_idx] = fold
    validation_seen[valid_idx] += 1
assert np.all(validation_seen == 1)
assert set(fold_assignment.tolist()) == set(range(1, N_SPLITS + 1))
folds = pd.DataFrame({"id": train_labels["id"], "fold": fold_assignment})
fold_profile = train_labels.assign(fold=fold_assignment).groupby("fold")["accepted"].agg(["size", "mean"])
fold_profile.columns = ["Lignes_validation", "Taux_approbation_validation"]
display_named(fold_profile)
