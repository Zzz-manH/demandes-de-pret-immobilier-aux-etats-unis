fixed_config(OUTPUT_DIR, {
    'purpose': 'learning curve (80% vs 100% of outer-train) + full-data refit of M1-M3 x 3 seeds',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'lc_group': LC_GROUP, 'lc_fraction': LC_FRACTION, 'full_rounds_scale': FULL_ROUNDS_SCALE,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def model_params(spec, iterations, seed):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations, random_seed=seed)


def early_stop_refit(spec, train, cats, idx, seed):
    """与五折模型（04）完全相同的协议：idx 内留 15% 早停 → 在整个 idx 上按 b/0.85 轮重训。"""
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
    require(int(model.tree_count_) == rounds, '重训轮数不符')
    return model, best, rounds
