lean_train, lean_test, lean_cats = variant_frames(LEAN)


def frames_for(spec):
    drop = spec.get('drop', [])
    unknown = set(drop) - set(lean_train.columns)
    require(not unknown, f'要删除的列不存在：{sorted(unknown)}')
    columns = [c for c in lean_train.columns if c not in drop]
    return lean_train[columns], [c for c in lean_cats if c not in drop]


def params_for(spec, **extra):
    params = dict(SCREEN_BASE)
    params.update({k: v for k, v in spec.items() if k != 'drop'})
    params.update(extra)
    return params


# 预检：每个变体在 3,000 行上训练 20 轮，几秒内暴露 GPU 不支持的参数组合，避免跑了一小时才报错
rng = np.random.default_rng(SEED)
probe_rows = rng.choice(len(y), 3000, replace=False)
failed = {}
for name, spec in VARIANTS.items():
    train, cats = frames_for(spec)
    try:
        model = catboost.CatBoostClassifier(**params_for(spec, iterations=20))
        model.fit(train.iloc[probe_rows[:2500]], y.iloc[probe_rows[:2500]], cat_features=cats,
                  eval_set=(train.iloc[probe_rows[2500:]], y.iloc[probe_rows[2500:]]))
        checked_probabilities(model, train.iloc[probe_rows[2500:]])
        print(f'预检通过：{name}（{train.shape[1]} 列）')
    except Exception as exc:
        failed[name] = repr(exc)
        print(f'预检失败，跳过：{name} → {exc!r}')
require('V0_base' not in failed, '基线 V0_base 预检失败，请检查环境')
