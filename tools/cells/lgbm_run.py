try:
    import lightgbm as lgb
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm'])
    import lightgbm as lgb
print('LightGBM', lgb.__version__)

lgb_train, lgb_test, lgb_cats = variant_frames(LGBM_BLOCKS)
for column in lgb_cats:               # 训练+测试的类别并集作为固定词表（不涉及标签）
    dtype = pd.CategoricalDtype(sorted(set(lgb_train[column]) | set(lgb_test[column])))
    lgb_train[column] = lgb_train[column].astype(dtype)
    lgb_test[column] = lgb_test[column].astype(dtype)
lgb_train.columns = [c.replace('-', '_') for c in lgb_train.columns]
lgb_test.columns = lgb_train.columns


def run_lgbm(name, config, fold):
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    inner_fit, inner_stop = train_test_split(
        fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED, stratify=y.iloc[fit_idx])
    probe = lgb.LGBMClassifier(**LGBM_BASE, **config)
    probe.fit(lgb_train.iloc[inner_fit], y.iloc[inner_fit],
              eval_set=[(lgb_train.iloc[inner_stop], y.iloc[inner_stop])], eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(OD_WAIT, verbose=False)])
    best_iteration = int(probe.best_iteration_)
    rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
    model = lgb.LGBMClassifier(**dict(LGBM_BASE, n_estimators=rounds), **config)
    model.fit(lgb_train.iloc[fit_idx], y.iloc[fit_idx])
    valid_prob = checked_probabilities(model, lgb_train.iloc[valid_idx])
    test_prob = checked_probabilities(model, lgb_test)
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'{name}__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': test_prob}).to_csv(
        f'{stem}__test.csv', index=False, float_format='%.12f')
    result = {
        'group': name, 'fold': fold,
        'definition': {'model': 'lightgbm', 'config': config, 'blocks': LGBM_BLOCKS},
        'feature_count': lgb_train.shape[1], 'inner_best_iteration': best_iteration,
        'early_stopped': best_iteration < MAX_ITERATIONS, 'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'lightgbm': lgb.__version__,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)
    return result


for fold in FOLDS:
    for name, config in LGBM_CONFIGS.items():
        path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            require(saved['definition']['config'] == config, f'{name} 的定义已改变；请改名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        result = run_lgbm(name, config, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  轮数={result["refit_rounds"]}  '
              f'{result["seconds"]/60:.1f} 分钟', flush=True)
