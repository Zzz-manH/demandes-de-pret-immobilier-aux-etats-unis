reference_oof = read_oof(OUTPUT_DIR, REFERENCE)
probability, actual = reference_oof['prob'], reference_oof['accepted']
bins = pd.qcut(probability, 20, duplicates='drop')
reliability = reference_oof.groupby(bins, observed=True).agg(
    Lignes=('prob', 'size'), Probabilité_moyenne=('prob', 'mean'), Taux_observé=('accepted', 'mean'))
reliability['Écart'] = reliability['Taux_observé'] - reliability['Probabilité_moyenne']
ece = float((reliability['Écart'].abs() * reliability['Lignes']).sum() / len(reference_oof))
base_rate = actual.mean()
summary = pd.DataFrame({
    'Modèle': {'AUC': roc_auc_score(actual, probability), 'Brier': brier_score_loss(actual, probability),
               'Log-loss': log_loss(actual, probability)},
    'Taux moyen pour tous': {'AUC': 0.5, 'Brier': base_rate * (1 - base_rate),
                             'Log-loss': log_loss(actual, np.full(len(actual), base_rate))},
})
display_named(summary.round(4))
print(f'Erreur de calibration (ECE, 20 tranches) : {ece:.4f} ; écart maximal dans une tranche : '
      f'{reliability["Écart"].abs().max():.4f}.')
display_named(reliability.reset_index(drop=True).round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot([0, 1], [0, 1], color='black', lw=0.8)
axes[0].plot(reliability['Probabilité_moyenne'], reliability['Taux_observé'], marker='o', ms=4, color='#4c72b0')
axes[0].set_xlabel('Probabilité prédite (moyenne de la tranche)')
axes[0].set_ylabel('Taux d’acceptation observé')
axes[0].set_title('Courbe de fiabilité, prédictions hors pli')
axes[1].hist([probability[actual.eq(1)], probability[actual.eq(0)]], bins=40, stacked=True,
             color=['#4c72b0', '#c44e52'], label=['acceptées', 'refusées'])
axes[1].set_yscale('log')
axes[1].set_xlabel('Probabilité prédite')
axes[1].set_ylabel('Lignes (échelle log)')
axes[1].legend()
plt.tight_layout()
plt.show()
