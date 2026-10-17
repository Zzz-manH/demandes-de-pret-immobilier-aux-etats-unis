results = load_results()
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='candidate', values='valid_auc')
require('B0_R2_fixed1600' in auc.columns, '缺少对照 B0_R2_fixed1600')
delta = auc.sub(auc['B0_R2_fixed1600'], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    'mean_delta_vs_B0': delta.mean(),
    'folds_better_than_B0': (delta > 0).sum(),
    'mean_refit_rounds': results.groupby('candidate')['refit_rounds'].mean(),
    'mean_logloss': results.groupby('candidate')['valid_logloss'].mean(),
    'total_minutes': results.groupby('candidate')['seconds'].sum() / 60,
}).sort_values('mean_delta_vs_B0', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
