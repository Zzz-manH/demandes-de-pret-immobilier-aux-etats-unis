fixed_config(OUTPUT_DIR, {
    'purpose': 'feature block ablation, fast GPU CatBoost, paired vs F0_base',
    'seed': SEED,
    'fast_params': {k: v for k, v in FAST_PARAMS.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})

selected = {n: b for n, b in VARIANTS.items() if RUN_ONLY is None or n in RUN_ONLY}
for fold in FOLDS:
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    for name, blocks in selected.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['blocks'] == blocks, f'{name} 的定义已改变；请改变体名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        train, _, cats = variant_frames(blocks)
        started = time.perf_counter()
        model = catboost.CatBoostClassifier(**FAST_PARAMS)
        model.fit(train.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats)
        require(int(model.tree_count_) == FAST_PARAMS['iterations'], '轮数不符')
        probability = checked_probabilities(model, train.iloc[valid_idx])
        actual = y.iloc[valid_idx]
        pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                      'prob': probability}).to_csv(
            OUTPUT_DIR / f'{name}__fold{fold}.csv', index=False, float_format='%.12f')
        result = {
            'variant': name, 'blocks': blocks, 'fold': fold,
            'feature_count': train.shape[1],
            'valid_auc': roc_auc_score(actual, probability),
            'valid_logloss': log_loss(actual, probability, labels=[0, 1]),
            'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        write_json(result_path, result)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'{train.shape[1]} 列  {result["seconds"]/60:.1f} 分钟', flush=True)
        del model, train
        gc.collect()
