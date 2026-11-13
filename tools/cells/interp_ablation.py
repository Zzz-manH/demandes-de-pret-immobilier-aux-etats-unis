ablation_runs = load_runs(OUTPUT_DIR)
require(len(ablation_runs) == 10, f'{len(ablation_runs)} entraînements trouvés, 10 attendus')
ablation_table, ablation_delta = paired_table(ablation_runs, 'group', REFERENCE, min_gain=0.0005)
per_fold = ablation_runs.pivot(index='fold', columns='group', values='valid_auc')[list(VARIANTS)]
per_fold['Écart'] = per_fold[WITHOUT_PROTECTED] - per_fold[REFERENCE]
display_named(per_fold.rename(columns=VARIANT_LABELS).round(5))
display_named(ablation_table.rename(index=VARIANT_LABELS).round(5))
print('Corrélation de Spearman entre les deux séries de prédictions hors pli : '
      f'{groups_frame["prob"].corr(groups_frame["prob_sans"], method="spearman"):.4f} ; '
      f'écart absolu moyen : {(groups_frame["prob"] - groups_frame["prob_sans"]).abs().mean():.4f}.')

spread = pd.DataFrame({
    label: {
        'Taux observé': np.ptp(GROUP_TABLES[column]['Taux observé']),
        'Probabilité moyenne, référence': np.ptp(GROUP_TABLES[column]['Probabilité moyenne']),
        'Probabilité moyenne, sans': np.ptp(GROUP_TABLES[column]['Probabilité moyenne, sans']),
    } for column, label in GROUP_COLUMNS.items()}).T
print('\nÉcart entre le groupe le plus accepté et le moins accepté')
display_named(spread.round(4))
for column, label in GROUP_COLUMNS.items():
    print(f'\n{label} ({column})')
    display_named(GROUP_TABLES[column][['Lignes', 'Taux observé', 'Probabilité moyenne',
                                        'Probabilité moyenne, sans', 'AUC', 'AUC, sans']].round(4))
