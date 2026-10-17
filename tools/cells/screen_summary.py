results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='variant', values='valid_auc')
require('V0_base' in auc.columns, '缺少基线 V0_base')
delta = auc.sub(auc['V0_base'], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    'mean_delta_vs_V0': delta.mean(),
    'folds_better_than_V0': (delta > 0).sum(),
    'mean_best_iteration': results.groupby('variant')['best_iteration'].mean(),
    'total_minutes': results.groupby('variant')['seconds'].sum() / 60,
})
table['keep'] = ((table['folds_done'] == 5) & (table['mean_delta_vs_V0'] >= 0.0005)
                 & (table['folds_better_than_V0'] >= 4))
table.loc['V0_base', 'keep'] = False
table = table.sort_values('mean_delta_vs_V0', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
if failed:
    print('预检失败、未运行的变体：', failed)
kept = table.index[table['keep']].tolist()
print('达到保留标准（≥ +0.0005 且 ≥ 4/5 折）的变体：', kept if kept else '无')
