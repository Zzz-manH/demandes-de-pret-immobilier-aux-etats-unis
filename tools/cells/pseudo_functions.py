fixed_config(OUTPUT_DIR, {
    'purpose': 'soft pseudo-labelling of test rows (CrossEntropy), fold-honest validation + full-data students',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'pl_group': PL_GROUP, 'pl_weight': PL_WEIGHT,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def model_params(spec, iterations, seed, loss='Logloss'):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations, random_seed=seed, loss_function=loss)


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


def pseudo_label_refit(spec, train, test, cats, idx, pseudo, seed):
    """真实行（idx）+ 全部测试行（软标签 pseudo）一起训练。早停集只取真实行的 15%，
    重训轮数按有效样本量（真实行 + 权重 × 伪标签行）等比放大。"""
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
    require(int(model.tree_count_) == rounds, '伪标签模型轮数不符')
    return model, best, rounds


def mean_test(paths):
    frames = [pd.read_csv(p, dtype={'id': str}) for p in paths]
    for f in frames:
        require(f['id'].equals(sample_submission['id']), f'测试 ID 顺序不一致：{paths[0]}')
    return np.mean([f['accepted'].to_numpy() for f in frames], axis=0)


def teacher_for_fold(fold):
    """第 k 折的老师 = 04 中只在第 k 折外层训练部分上训练的 9 个模型对测试集的平均预测。"""
    return mean_test([CATBOOST_DIR / f'{b}{s}__fold{fold}__test.csv'
                      for b in CAT_BASES for s in SEED_SUFFIXES.values()])


def teacher_valid(fold):
    frames = [pd.read_csv(CATBOOST_DIR / f'{b}{s}__fold{fold}__valid.csv')
              for b in CAT_BASES for s in SEED_SUFFIXES.values()]
    for f in frames[1:]:
        require(f['id'].equals(frames[0]['id']), '老师 OOF 行顺序不一致')
    return frames[0][['id']].assign(teacher=np.mean([f['prob'].to_numpy() for f in frames], axis=0))
