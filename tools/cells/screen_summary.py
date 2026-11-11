results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
require(not results.empty, 'Aucun résultat terminé')
auc = results.pivot(index='fold', columns='variant', values='valid_auc')
require(TRAINING_STANDARD in auc.columns, 'Référence Boosting standard manquante')
delta = auc.sub(auc[TRAINING_STANDARD], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    MEAN_GAIN_VS_STANDARD_BOOSTING: delta.mean(),
    FOLDS_BETTER_THAN_STANDARD_BOOSTING: (delta > 0).sum(),
    'mean_best_iteration': results.groupby('variant')['best_iteration'].mean(),
    'total_minutes': results.groupby('variant')['seconds'].sum() / 60,
})
table['keep'] = ((table['folds_done'] == 5) & (table[MEAN_GAIN_VS_STANDARD_BOOSTING] >= 0.0005)
                 & (table[FOLDS_BETTER_THAN_STANDARD_BOOSTING] >= 4))
table.loc[TRAINING_STANDARD, 'keep'] = False
table = table.sort_values(MEAN_GAIN_VS_STANDARD_BOOSTING, ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display_named(table.round(6))
display_named(delta.round(6))
if failed:
    print('Variantes exclues après échec du précontrôle :', failed)
kept = table.index[table['keep']].tolist()
print('Variantes retenues (gain ≥ +0.0005 sur ≥ 4/5 folds) :', kept if kept else 'Aucune')
