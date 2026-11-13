fixed_config(OUTPUT_DIR, {
    'purpose': 'reference vs model without protected columns, 5-fold with inner early stop; SHAP values on one fold',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'params': MODEL_PARAMS, 'blocks': MODEL_BLOCKS, 'variants': VARIANTS,
    'shap_fold': SHAP_FOLD, 'shap_rows': SHAP_ROWS,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})
require(BLOCKS is not None, 'Feature store absent : exécuter 02_feature_engineering.ipynb')


def run_one(name, dropped, fold):
    """Même protocole que le notebook 04 : arrêt anticipé interne, puis réentraînement pendant b/0.85 itérations."""
    train = full_train.drop(columns=dropped)
    cats = [c for c in full_cats if c not in dropped]
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    inner_fit, inner_stop = train_test_split(
        fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED, stratify=y.iloc[fit_idx],
    )
    probe = catboost.CatBoostClassifier(**GPU_BASE, **MODEL_PARAMS, iterations=MAX_ITERATIONS,
                                        eval_metric='Logloss', od_type='Iter',
                                        od_wait=OD_WAIT, use_best_model=True)
    probe.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
              eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
    best_iteration = int(probe.get_best_iteration()) + 1
    rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
    del probe
    gc.collect()
    model = catboost.CatBoostClassifier(**GPU_BASE, **MODEL_PARAMS, iterations=rounds, use_best_model=False)
    model.fit(train.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats)
    require(int(model.tree_count_) == rounds, 'Nombre d’itérations incorrect pour le réentraînement externe')
    valid_prob = checked_probabilities(model, train.iloc[valid_idx])
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'{name}__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    if name == REFERENCE and fold == SHAP_FOLD:
        rows = np.sort(np.random.default_rng(SEED).choice(valid_idx, size=SHAP_ROWS, replace=False))
        values = model.get_feature_importance(
            catboost.Pool(train.iloc[rows], cat_features=cats), type='ShapValues')
        require(values.shape == (SHAP_ROWS, train.shape[1] + 1), 'Dimensions des valeurs de Shapley incorrectes')
        shap_frame = pd.DataFrame(values[:, :-1], columns=train.columns)
        shap_frame.insert(0, 'expected_value', values[:, -1])
        shap_frame.insert(0, 'id', train_ids.iloc[rows].to_numpy())
        shap_frame.to_parquet(SHAP_PATH, index=False)
    result = {
        'group': name, 'fold': fold, 'dropped': dropped, 'feature_count': train.shape[1],
        'inner_best_iteration': best_iteration, 'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)   # Écrire le JSON en dernier : il marque la fin du pli
    del model, train
    gc.collect()
    return result


for fold in FOLDS:      # les deux variantes d'un pli se suivent dans la même session
    for name, dropped in VARIANTS.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['dropped'] == dropped, f'{name} : définition modifiée ; changer EXPERIMENT_ID')
            print(f'{name} fold {fold}: terminé, AUC={saved["valid_auc"]:.6f}, ignoré')
            continue
        result = run_one(name, dropped, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'itérations={result["refit_rounds"]}  {result["seconds"]/60:.1f} min', flush=True)
