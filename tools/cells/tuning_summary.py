results = load_results()
require(not results.empty, 'Aucun résultat terminé')
auc = results.pivot(index='fold', columns='candidate', values='valid_auc')
require(TUNING_DEPTH8_FIXED1600 in auc.columns, 'Référence Profondeur 8 · 1 600 itérations fixes manquante')
delta = auc.sub(auc[TUNING_DEPTH8_FIXED1600], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    MEAN_GAIN_VS_FIXED_ITERATIONS: delta.mean(),
    FOLDS_BETTER_THAN_FIXED_ITERATIONS: (delta > 0).sum(),
    'mean_refit_rounds': results.groupby('candidate')['refit_rounds'].mean(),
    'mean_logloss': results.groupby('candidate')['valid_logloss'].mean(),
    'total_minutes': results.groupby('candidate')['seconds'].sum() / 60,
}).sort_values(MEAN_GAIN_VS_FIXED_ITERATIONS, ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display_named(table.round(6))
display_named(delta.round(6))
