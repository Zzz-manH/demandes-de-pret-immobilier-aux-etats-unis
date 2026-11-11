nn_done = sorted(json.loads(p.read_text(encoding='utf-8'))['fold']
                 for p in OUTPUT_DIR.glob('TabMMini__fold?.json'))
require(nn_done, 'Aucun fold du réseau terminé')

FINAL_MODEL_NAMES = [MODEL_DEPTH6_ENRICHED, MODEL_DEPTH8_ENRICHED, MODEL_DEPTH6_REDUCED]
SEED_SUFFIXES = ['', '_s2027', '_s31415']
cat_groups = [b + s for b in FINAL_MODEL_NAMES for s in SEED_SUFFIXES]
missing = [g for g in cat_groups for k in range(1, 6)
           if not (CATBOOST_DIR / f'{g}__fold{k}__valid.csv').is_file()]
require(not missing, f'Résultats CatBoost de 04 manquants, par exemple {missing[:3]} ; terminer d’abord 04')


def catboost_fold(fold):
    'l’ensemble multigraines de 04 : moyenne des 3 seeds de chaque groupe, puis moyenne des 3 groupes à parts égales.'
    per_base = []
    for base in FINAL_MODEL_NAMES:
        frames = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{fold}__valid.csv') for s in SEED_SUFFIXES]
        for f in frames[1:]:
            require(f['id'].equals(frames[0]['id']), 'Ordre des lignes OOF CatBoost incohérent')
        per_base.append(np.mean([f['prob'].to_numpy() for f in frames], axis=0))
    return frames[0][['id', 'accepted']].assign(cat=np.mean(per_base, axis=0))


rows = []
for fold in nn_done:
    nn_valid = pd.read_csv(OUTPUT_DIR / f'TabMMini__fold{fold}__valid.csv')
    merged = catboost_fold(fold).merge(nn_valid[['id', 'prob']], on='id', validate='one_to_one')
    require(len(merged) == len(nn_valid), 'Lignes de validation différentes entre réseau et CatBoost')
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
display_named(table.round(6))
summary = table.mean().to_frame('mean').T
display_named(summary.round(6))
table.to_csv(OUTPUT_DIR / 'blend_by_fold.csv')

gap = summary['catboost_auc'].iloc[0] - summary['tabm_auc'].iloc[0]
best_w = max(BLEND_WEIGHTS, key=lambda w: summary[f'delta_w{int(w * 100)}'].iloc[0])
best_delta = summary[f'delta_w{int(best_w * 100)}'].iloc[0]
print(f'TabM seul est inférieur à CatBoost de {gap:.5f} ; corrélation avec CatBoost {summary["corr"].iloc[0]:.4f} ; '
      f'Meilleur poids de fusion {best_w}: {best_delta:+.6f}')
if len(nn_done) < 5:
    verdict = ('Poursuivre sur cinq folds : régler FOLDS sur [1, 2, 3, 4, 5] et relancer ; le fold 1 terminé sera ignoré.'
               if best_delta >= 0.0003 else
               'Gain de fusion inférieur à 0.0003 sur le fold 1 : arrêter ici et conserver l’ensemble multigraines de CatBoost.')
    print('Bilan : ' + verdict)
else:
    better = int((table[f'delta_w{int(best_w * 100)}'] > 0).sum())
    print(f'Cinq folds : poids {best_w}, amélioration sur {better}/5 folds face à CatBoost.')
    cat_test = []
    for base in FINAL_MODEL_NAMES:
        tests = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{k}__test.csv', dtype={'id': str})
                 for s in SEED_SUFFIXES for k in range(1, 6)]
        for t in tests:
            require(t['id'].equals(sample_submission['id']), 'Ordre des ID test CatBoost incohérent')
        cat_test.append(np.mean([t['accepted'].to_numpy() for t in tests], axis=0))
    cat_test = np.mean(cat_test, axis=0)
    nn_tests = [pd.read_csv(OUTPUT_DIR / f'TabMMini__fold{k}__test.csv', dtype={'id': str})
                for k in range(1, 6)]
    for t in nn_tests:
        require(t['id'].equals(sample_submission['id']), 'Ordre des ID test TabM incohérent')
    nn_test = np.mean([t['accepted'].to_numpy() for t in nn_tests], axis=0)
    for w in BLEND_WEIGHTS:
        path = OUTPUT_DIR / f'{TABM_ENSEMBLE_SUBMISSION_PREFIX}{int(w * 100)}.csv'
        pd.DataFrame({'id': sample_submission['id'],
                      'accepted': (1 - w) * cat_test + w * nn_test}).to_csv(
            path, index=False, float_format='%.12f')
        print('Fichier écrit :', path.name)
    print('Remplacer l’ensemble multigraines seulement si un poids gagne au moins +0.0003 en moyenne, avec un progrès sur au moins 4/5 folds.')
