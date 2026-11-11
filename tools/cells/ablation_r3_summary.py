results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))])
require(not results.empty, 'Aucun résultat terminé')
auc = results.pivot(index='fold', columns='variant', values='valid_auc')
require(ABLATION_ENRICHED_REFERENCE in auc.columns, 'Référence Référence : variables enrichies manquante')
delta = auc.sub(auc[ABLATION_ENRICHED_REFERENCE], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    MEAN_GAIN_VS_ENRICHED_FEATURES: delta.mean(),
    FOLDS_BETTER_THAN_ENRICHED_FEATURES: (delta > 0).sum(),
    'mean_logloss': results.groupby('variant')['valid_logloss'].mean(),
    'feature_count': results.groupby('variant')['feature_count'].first(),
})
table['keep'] = ((table['folds_done'] == 5) & (table[MEAN_GAIN_VS_ENRICHED_FEATURES] >= 0.0005)
                 & (table[FOLDS_BETTER_THAN_ENRICHED_FEATURES] >= 4))
table.loc[ABLATION_ENRICHED_REFERENCE, 'keep'] = False
table = table.sort_values(MEAN_GAIN_VS_ENRICHED_FEATURES, ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display_named(table.round(6))
display_named(delta.round(6))
