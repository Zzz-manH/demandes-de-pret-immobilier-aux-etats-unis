def fold_average_bseeds():
    per_base = []
    for base in CAT_BASES:
        tests = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{k}__test.csv', dtype={'id': str})
                 for s in SEED_SUFFIXES.values() for k in range(1, 6)]
        for t in tests:
            require(t['id'].equals(sample_submission['id']), '折模型测试 ID 顺序不一致')
        per_base.append(np.mean([t['accepted'].to_numpy() for t in tests], axis=0))
    return np.mean(per_base, axis=0)


full_paths = {(b, s): OUTPUT_DIR / f'FULL_{b}_seed{s}__test.csv' for b in CAT_BASES for s in SEED_SUFFIXES}
missing = [p.name for p in full_paths.values() if not p.is_file()]
if missing:
    print(f'全量模型还缺 {len(missing)} 个（例如 {missing[0]}）；完成后再运行本格。')
else:
    per_base = []
    for base in CAT_BASES:
        frames = [pd.read_csv(full_paths[(base, s)], dtype={'id': str}) for s in SEED_SUFFIXES]
        for f in frames:
            require(f['id'].equals(sample_submission['id']), '全量模型测试 ID 顺序不一致')
        per_base.append(np.mean([f['accepted'].to_numpy() for f in frames], axis=0))
    full = np.mean(per_base, axis=0)
    fold_avg = fold_average_bseeds()
    print(f'全量融合 与 折平均 B_seeds 的测试预测相关系数：{np.corrcoef(full, fold_avg)[0, 1]:.5f}')
    for name, probability in [('full_Bseeds', full), ('hybrid_50_50', 0.5 * full + 0.5 * fold_avg)]:
        require(np.isfinite(probability).all() and len(probability) == 45000, f'{name} 概率无效')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
            path, index=False, float_format='%.12f')
        print('写出', path.name)
    if 'curve' in globals() and 'gain_84k_vs_67k' in curve and curve['gain_84k_vs_67k'].notna().sum() == 5:
        gain = curve['gain_84k_vs_67k']
        if gain.mean() >= 0.0005 and (gain > 0).sum() >= 4:
            advice = '学习曲线支持全量重训：最终主选 submission_hybrid_50_50.csv，备选保留折平均的 B_seeds。'
        elif gain.mean() >= 0.0002:
            advice = '学习曲线收益偏小：hybrid 可以作为备选，主选仍为折平均的 B_seeds。'
        else:
            advice = '学习曲线没有显示数据量收益：保持折平均的 B_seeds，不使用全量模型。'
        print('判断：' + advice)
