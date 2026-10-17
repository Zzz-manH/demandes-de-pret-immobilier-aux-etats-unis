results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))])
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='variant', values='valid_auc')
require('H0_G7' in auc.columns, '缺少基线 H0_G7')
delta = auc.sub(auc['H0_G7'], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    'mean_delta_vs_H0': delta.mean(),
    'folds_better_than_H0': (delta > 0).sum(),
    'mean_logloss': results.groupby('variant')['valid_logloss'].mean(),
    'feature_count': results.groupby('variant')['feature_count'].first(),
})
table['keep'] = ((table['folds_done'] == 5) & (table['mean_delta_vs_H0'] >= 0.0005)
                 & (table['folds_better_than_H0'] >= 4))
table.loc['H0_G7', 'keep'] = False
table = table.sort_values('mean_delta_vs_H0', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
