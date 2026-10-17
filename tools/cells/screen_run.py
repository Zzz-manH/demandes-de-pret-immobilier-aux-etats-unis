fixed_config(OUTPUT_DIR, {
    'purpose': 'data-efficiency screen on M3 features; inner 15% early stop, score probe model on outer fold',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'screen_base': {k: v for k, v in SCREEN_BASE.items() if k != 'thread_count'},
    'features': LEAN, 'data_hashes': DATA_HASHES,
})
selected = {n: s for n, s in VARIANTS.items()
            if (RUN_ONLY is None or n in RUN_ONLY) and n not in failed}
for fold in FOLDS:
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION,
                                             random_state=SEED, stratify=y.iloc[fit_idx])
    for name, spec in selected.items():
        path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            require(saved['spec'] == spec, f'{name} 的定义已改变；请改变体名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        train, cats = frames_for(spec)
        started = time.perf_counter()
        model = catboost.CatBoostClassifier(**params_for(spec))
        model.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
                  eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
        best = int(model.get_best_iteration()) + 1
        probability = checked_probabilities(model, train.iloc[valid_idx])
        actual = y.iloc[valid_idx]
        pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                      'prob': probability}).to_csv(OUTPUT_DIR / f'{name}__fold{fold}__valid.csv',
                                                   index=False, float_format='%.12f')
        result = {'variant': name, 'fold': fold, 'spec': spec, 'feature_count': train.shape[1],
                  'best_iteration': best,
                  'hit_iteration_limit': best >= SCREEN_BASE['iterations'] - SCREEN_BASE['od_wait'],
                  'valid_auc': roc_auc_score(actual, probability),
                  'valid_logloss': log_loss(actual, probability, labels=[0, 1]),
                  'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                  'finished_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(path, result)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  最佳轮数={best}  '
              f'{result["seconds"]/60:.1f} 分钟', flush=True)
        del model
        gc.collect()
