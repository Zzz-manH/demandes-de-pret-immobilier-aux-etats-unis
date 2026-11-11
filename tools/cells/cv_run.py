fixed_config(OUTPUT_DIR, {
    'purpose': 'params x feature blocks, 5-fold with inner early stop; fold-averaged test predictions',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def model_params(spec, iterations):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations)


def run_one(name, group, fold):
    spec = group['params']
    train, test, cats = variant_frames(group['blocks'])
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    best_iteration, early_stopped = None, None
    if spec.get('early_stop', True):
        inner_fit, inner_stop = train_test_split(
            fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED,
            stratify=y.iloc[fit_idx],
        )
        probe = catboost.CatBoostClassifier(**model_params(spec, MAX_ITERATIONS),
                                            eval_metric='Logloss', od_type='Iter',
                                            od_wait=OD_WAIT, use_best_model=True)
        probe.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
                  eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
        best_iteration = int(probe.get_best_iteration()) + 1
        early_stopped = int(probe.tree_count_) < MAX_ITERATIONS
        if not early_stopped:
            print(f'  Attention : {name} atteint {MAX_ITERATIONS} itérations sans arrêt anticipé')
        rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
        del probe
        gc.collect()
    else:
        rounds = int(spec['iterations'])
    model = catboost.CatBoostClassifier(**model_params(spec, rounds), use_best_model=False)
    model.fit(train.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats)
    require(int(model.tree_count_) == rounds, 'Nombre d’itérations incorrect pour le réentraînement externe')
    valid_prob = checked_probabilities(model, train.iloc[valid_idx])
    test_prob = checked_probabilities(model, test)
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'{name}__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': test_prob}).to_csv(
        f'{stem}__test.csv', index=False, float_format='%.12f')
    result = {
        'group': name, 'fold': fold, 'definition': group,
        'feature_count': train.shape[1],
        'inner_best_iteration': best_iteration, 'early_stopped': early_stopped,
        'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)   # Écrire le JSON en dernier : il marque la fin du fold
    del model, train, test
    gc.collect()
    return result


selected = {n: g for n, g in GROUPS.items() if RUN_ONLY is None or n in RUN_ONLY}
for fold in FOLDS:
    for name, group in selected.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['definition'] == group, f'{name} : définition modifiée ; renommer le groupe')
            print(f'{name} fold {fold}: terminé, AUC={saved["valid_auc"]:.6f}, ignoré')
            continue
        result = run_one(name, group, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'itérations={result["refit_rounds"]}  {result["seconds"]/60:.1f} min', flush=True)
