results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
done = {g for g, part in results.groupby('group') if sorted(part['fold']) == [1, 2, 3, 4, 5]}
print('Groupes ayant terminé les cinq folds :', sorted(done))


def load(name):
    valid = pd.concat([pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__valid.csv').assign(fold=k)
                       for k in range(1, 6)]).sort_values('id', kind='stable').reset_index(drop=True)
    tests = [pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__test.csv', dtype={'id': str}) for k in range(1, 6)]
    for frame in tests:
        require(frame['id'].equals(sample_submission['id']), f'{name} : ordre des ID test incohérent')
    return valid, np.mean([t['accepted'].to_numpy() for t in tests], axis=0)


loaded = {name: load(name) for name in sorted(done)}
reference = loaded[FINAL_MODEL_NAMES[0]][0][['id', 'accepted', 'fold']]
for name, (valid, _) in loaded.items():
    require(valid['id'].equals(reference['id']), f'{name} : ID OOF différents de la référence')


def mean_of(names):
    return (np.mean([loaded[n][0]['prob'].to_numpy() for n in names], axis=0),
            np.mean([loaded[n][1] for n in names], axis=0))


def fold_aucs(probability):
    return pd.Series({k: roc_auc_score(reference.loc[reference['fold'].eq(k), 'accepted'],
                                       probability[reference['fold'].eq(k).to_numpy()])
                      for k in range(1, 6)})


blends = {}
require(all(b in done for b in FINAL_MODEL_NAMES), 'Résultats sur cinq folds manquants pour les trois modèles finaux')
blends[ENSEMBLE_THREE_MODELS_SEED42] = mean_of(FINAL_MODEL_NAMES)                       # = soumission ayant obtenu 0.92190
per_base = [mean_of([b] + [f'{b}_s{s}' for s in SEEDS_EXTRA if f'{b}_s{s}' in done]) for b in FINAL_MODEL_NAMES]
n_seeds = {b: 1 + sum(f'{b}_s{s}' in done for s in SEEDS_EXTRA) for b in FINAL_MODEL_NAMES}
seed_tag = ENSEMBLE_MULTISEED_PREFIX + '_'.join(str(n_seeds[b]) for b in FINAL_MODEL_NAMES)
blends[seed_tag] = (np.mean([v for v, _ in per_base], axis=0), np.mean([t for _, t in per_base], axis=0))

ref_folds = fold_aucs(blends[ENSEMBLE_THREE_MODELS_SEED42][0])
rows = []
for name, (valid_prob, test_prob) in blends.items():
    folds = fold_aucs(valid_prob)
    require(np.isfinite(test_prob).all() and len(test_prob) == 45000, f'{name} : probabilités test invalides')
    path = OUTPUT_DIR / f'submission_{name}.csv'
    pd.DataFrame({'id': sample_submission['id'], 'accepted': test_prob}).to_csv(
        path, index=False, float_format='%.12f')
    rows.append({'blend': name, 'mean_auc': folds.mean(),
                 GAIN_VS_THREE_MODEL_ENSEMBLE: (folds - ref_folds).mean(),
                 FOLDS_BETTER_THAN_THREE_MODEL_ENSEMBLE: int(((folds - ref_folds) > 0).sum()),
                 'oof_auc': roc_auc_score(reference['accepted'], valid_prob),
                 'submission': path.name})
table = pd.DataFrame(rows).set_index('blend').sort_values('mean_auc', ascending=False)
table.to_csv(OUTPUT_DIR / 'blend_summary.csv')
display_named(table.round(6))
singles = results.groupby('group')['valid_auc'].agg(['count', 'mean']).round(6)
display_named(singles.sort_values('mean', ascending=False))
best = table['mean_auc'].idxmax()
print(f'Meilleure moyenne sur cinq folds : {best}({table.loc[best, "mean_auc"]:.6f}, écart à la soumission précédente {table.loc[best, GAIN_VS_THREE_MODEL_ENSEMBLE]:+.6f})'
      f' → {table.loc[best, "submission"]}')
