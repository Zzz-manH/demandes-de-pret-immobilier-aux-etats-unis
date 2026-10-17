fixed_config(OUTPUT_DIR, {
    'purpose': 'GPU CatBoost screening, inner 15% early stop then outer refit',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def candidate_params(spec, iterations):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'dti', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations)


def run_one(name, spec, fold):
    features = X_dti if spec.get('dti') else X_base
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    best_iteration = None
    early_stopped = None
    if spec.get('early_stop', True):
        inner_fit, inner_stop = train_test_split(
            fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED,
            stratify=y.iloc[fit_idx],
        )
        probe = catboost.CatBoostClassifier(**candidate_params(spec, MAX_ITERATIONS),
                                            eval_metric='Logloss', od_type='Iter',
                                            od_wait=OD_WAIT, use_best_model=True)
        probe.fit(features.iloc[inner_fit], y.iloc[inner_fit],
                  cat_features=CATEGORICAL_COLUMNS,
                  eval_set=(features.iloc[inner_stop], y.iloc[inner_stop]))
        best_iteration = int(probe.get_best_iteration()) + 1
        early_stopped = int(probe.tree_count_) < MAX_ITERATIONS
        if not early_stopped:
            print(f'  注意：{name} 到 {MAX_ITERATIONS} 轮仍未早停，结果可能偏保守')
        rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
        del probe
        gc.collect()
    else:
        rounds = int(spec['iterations'])
    model = catboost.CatBoostClassifier(**candidate_params(spec, rounds), use_best_model=False)
    model.fit(features.iloc[fit_idx], y.iloc[fit_idx], cat_features=CATEGORICAL_COLUMNS)
    require(int(model.tree_count_) == rounds, '外层重训轮数不符')
    probability = checked_probabilities(model, features.iloc[valid_idx])
    actual = y.iloc[valid_idx]
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': probability}).to_csv(
        OUTPUT_DIR / f'{name}__fold{fold}.csv', index=False, float_format='%.12f')
    result = {
        'candidate': name, 'fold': fold,
        'spec': {k: v for k, v in spec.items()},
        'feature_count': features.shape[1],
        'inner_best_iteration': best_iteration, 'early_stopped': early_stopped, 'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, probability),
        'valid_logloss': log_loss(actual, probability, labels=[0, 1]),
        'valid_brier': brier_score_loss(actual, probability),
        'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUTPUT_DIR / f'{name}__fold{fold}.json', result)
    del model
    gc.collect()
    return result


def load_results():
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))]
    return pd.DataFrame(rows)


selected = {n: s for n, s in CANDIDATES.items() if RUN_ONLY is None or n in RUN_ONLY}
for fold in FOLDS:
    for name, spec in selected.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['spec'] == spec, f'{name} 的定义已改变；请改候选名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        result = run_one(name, spec, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'轮数={result["refit_rounds"]}  {result["seconds"]/60:.1f} 分钟', flush=True)
