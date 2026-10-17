nn_done = sorted(json.loads(p.read_text(encoding='utf-8'))['fold']
                 for p in OUTPUT_DIR.glob('TabMMini__fold?.json'))
require(nn_done, '神经网络还没有完成的折')

CAT_BASES = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']
SEED_SUFFIXES = ['', '_s2027', '_s31415']
cat_groups = [b + s for b in CAT_BASES for s in SEED_SUFFIXES]
missing = [g for g in cat_groups for k in range(1, 6)
           if not (CATBOOST_DIR / f'{g}__fold{k}__valid.csv').is_file()]
require(not missing, f'缺少 04 的 CatBoost 五折结果，例如 {missing[:3]}；请先完成 04')


def catboost_fold(fold):
    """04 的 B_seeds：每个基础组先对 3 个种子平均，再三组等权。"""
    per_base = []
    for base in CAT_BASES:
        frames = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{fold}__valid.csv') for s in SEED_SUFFIXES]
        for f in frames[1:]:
            require(f['id'].equals(frames[0]['id']), 'CatBoost OOF 行顺序不一致')
        per_base.append(np.mean([f['prob'].to_numpy() for f in frames], axis=0))
    return frames[0][['id', 'accepted']].assign(cat=np.mean(per_base, axis=0))


rows = []
for fold in nn_done:
    nn_valid = pd.read_csv(OUTPUT_DIR / f'TabMMini__fold{fold}__valid.csv')
    merged = catboost_fold(fold).merge(nn_valid[['id', 'prob']], on='id', validate='one_to_one')
    require(len(merged) == len(nn_valid), '神经网络与 CatBoost 的验证行不一致')
    row = {'fold': fold,
           'catboost_auc': roc_auc_score(merged['accepted'], merged['cat']),
           'tabm_auc': roc_auc_score(merged['accepted'], merged['prob']),
           'corr': np.corrcoef(merged['cat'], merged['prob'])[0, 1]}
    for w in BLEND_WEIGHTS:
        row[f'blend_w{int(w * 100)}'] = roc_auc_score(merged['accepted'],
                                                     (1 - w) * merged['cat'] + w * merged['prob'])
    rows.append(row)
table = pd.DataFrame(rows).set_index('fold')
for w in BLEND_WEIGHTS:
    table[f'delta_w{int(w * 100)}'] = table[f'blend_w{int(w * 100)}'] - table['catboost_auc']
display(table.round(6))
summary = table.mean().to_frame('mean').T
display(summary.round(6))
table.to_csv(OUTPUT_DIR / 'blend_by_fold.csv')

gap = summary['catboost_auc'].iloc[0] - summary['tabm_auc'].iloc[0]
best_w = max(BLEND_WEIGHTS, key=lambda w: summary[f'delta_w{int(w * 100)}'].iloc[0])
best_delta = summary[f'delta_w{int(best_w * 100)}'].iloc[0]
print(f'TabM 单独比 CatBoost 低 {gap:.5f}；与 CatBoost 的相关系数 {summary["corr"].iloc[0]:.4f}；'
      f'最好的融合权重 {best_w}：{best_delta:+.6f}')
if len(nn_done) < 5:
    verdict = ('值得跑完五折：把 FOLDS 改为 [1, 2, 3, 4, 5] 后从头运行（第 1 折会自动跳过）。'
               if best_delta >= 0.0003 else
               '第 1 折上融合提升不足 0.0003，建议到此为止，保持 CatBoost 的 B_seeds。')
    print('判断：' + verdict)
else:
    better = int((table[f'delta_w{int(best_w * 100)}'] > 0).sum())
    print(f'五折：权重 {best_w} 时 {better}/5 折优于 CatBoost。')
    cat_test = []
    for base in CAT_BASES:
        tests = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{k}__test.csv', dtype={'id': str})
                 for s in SEED_SUFFIXES for k in range(1, 6)]
        for t in tests:
            require(t['id'].equals(sample_submission['id']), 'CatBoost 测试 ID 顺序不一致')
        cat_test.append(np.mean([t['accepted'].to_numpy() for t in tests], axis=0))
    cat_test = np.mean(cat_test, axis=0)
    nn_tests = [pd.read_csv(OUTPUT_DIR / f'TabMMini__fold{k}__test.csv', dtype={'id': str})
                for k in range(1, 6)]
    for t in nn_tests:
        require(t['id'].equals(sample_submission['id']), 'TabM 测试 ID 顺序不一致')
    nn_test = np.mean([t['accepted'].to_numpy() for t in nn_tests], axis=0)
    for w in BLEND_WEIGHTS:
        path = OUTPUT_DIR / f'submission_Bseeds_TabM_w{int(w * 100)}.csv'
        pd.DataFrame({'id': sample_submission['id'],
                      'accepted': (1 - w) * cat_test + w * nn_test}).to_csv(
            path, index=False, float_format='%.12f')
        print('写出', path.name)
    print('只有当某个权重五折平均 ≥ +0.0003 且 ≥ 4/5 折更好时，才替换 B_seeds。')
