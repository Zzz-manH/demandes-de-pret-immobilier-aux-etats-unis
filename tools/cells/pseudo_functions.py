fixed_config(OUTPUT_DIR, {
    'purpose': 'soft pseudo-labelling of test rows (CrossEntropy), fold-honest validation + full-data students',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'pl_group': PSEUDO_LABEL_MODEL, 'pl_weight': PL_WEIGHT,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def model_params(spec, iterations, seed, loss='Logloss'):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations, random_seed=seed, loss_function=loss)


def early_stop_refit(spec, train, cats, idx, seed):
    """Protocole identique à 04 : réserver 15 % de idx pour l’arrêt anticipé, puis réentraîner sur tout idx pendant b/0.85 itérations."""
    inner_fit, inner_stop = train_test_split(idx, test_size=INNER_STOP_FRACTION,
                                             random_state=SEED, stratify=y.iloc[idx])
    probe = catboost.CatBoostClassifier(**model_params(spec, MAX_ITERATIONS, seed),
                                        eval_metric='Logloss', od_type='Iter',
                                        od_wait=OD_WAIT, use_best_model=True)
    probe.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
              eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
    best = int(probe.get_best_iteration()) + 1
    rounds = int(round(best / (1 - INNER_STOP_FRACTION)))
    del probe
    gc.collect()
    model = catboost.CatBoostClassifier(**model_params(spec, rounds, seed), use_best_model=False)
    model.fit(train.iloc[idx], y.iloc[idx], cat_features=cats)
    require(int(model.tree_count_) == rounds, 'Nombre d’itérations de réentraînement incorrect')
    return model, best, rounds


def pseudo_label_refit(spec, train, test, cats, idx, pseudo, seed):
    """Entraîner sur les lignes réelles (idx) et tout le test (pseudo-labels souples). Réserver 15 % des lignes réelles à l’arrêt anticipé, puis adapter les itérations au volume effectif : lignes réelles + poids × lignes pseudo-étiquetées."""
    inner_fit, inner_stop = train_test_split(idx, test_size=INNER_STOP_FRACTION,
                                             random_state=SEED, stratify=y.iloc[idx])

    def stacked(real_idx):
        frame = pd.concat([train.iloc[real_idx], test], ignore_index=True)
        target = np.concatenate([y.iloc[real_idx].to_numpy(dtype='float64'), pseudo])
        weight = np.concatenate([np.ones(len(real_idx)), np.full(len(test), PL_WEIGHT)])
        return frame, target, weight

    X_fit, y_fit, w_fit = stacked(inner_fit)
    probe = catboost.CatBoostClassifier(**model_params(spec, MAX_ITERATIONS, seed, loss='CrossEntropy'),
                                        eval_metric='CrossEntropy', od_type='Iter',
                                        od_wait=OD_WAIT, use_best_model=True)
    probe.fit(X_fit, y_fit, sample_weight=w_fit, cat_features=cats,
              eval_set=(train.iloc[inner_stop], y.iloc[inner_stop].astype('float64')))
    best = int(probe.get_best_iteration()) + 1
    effective_inner = len(inner_fit) + PL_WEIGHT * len(test)
    effective_full = len(idx) + PL_WEIGHT * len(test)
    rounds = int(round(best * effective_full / effective_inner))
    del probe, X_fit, y_fit, w_fit
    gc.collect()
    X_all, y_all, w_all = stacked(idx)
    model = catboost.CatBoostClassifier(**model_params(spec, rounds, seed, loss='CrossEntropy'),
                                        use_best_model=False)
    model.fit(X_all, y_all, sample_weight=w_all, cat_features=cats)
    require(int(model.tree_count_) == rounds, 'Nombre d’itérations incorrect pour le modèle à pseudo-labels')
    return model, best, rounds


def mean_test(paths):
    frames = [pd.read_csv(p, dtype={'id': str}) for p in paths]
    for f in frames:
        require(f['id'].equals(sample_submission['id']), f'Ordre des ID test incohérent : {paths[0]}')
    return np.mean([f['accepted'].to_numpy() for f in frames], axis=0)


def teacher_for_fold(fold):
    """Enseignant du fold k : moyenne test des 9 modèles de 04 entraînés uniquement sur le train externe du fold k."""
    return mean_test([CATBOOST_DIR / f'{b}{s}__fold{fold}__test.csv'
                      for b in FINAL_MODEL_NAMES for s in SEED_SUFFIXES.values()])


def teacher_valid(fold):
    frames = [pd.read_csv(CATBOOST_DIR / f'{b}{s}__fold{fold}__valid.csv')
              for b in FINAL_MODEL_NAMES for s in SEED_SUFFIXES.values()]
    for f in frames[1:]:
        require(f['id'].equals(frames[0]['id']), 'Ordre des lignes OOF de l’enseignant incohérent')
    return frames[0][['id']].assign(teacher=np.mean([f['prob'].to_numpy() for f in frames], axis=0))
