def fold_average_multiseed_ensemble():
    per_base = []
    for base in FINAL_MODEL_NAMES:
        tests = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{k}__test.csv', dtype={'id': str})
                 for s in SEED_SUFFIXES.values() for k in range(1, 6)]
        for t in tests:
            require(t['id'].equals(sample_submission['id']), 'Ordre des ID test incohérent entre modèles de folds')
        per_base.append(np.mean([t['accepted'].to_numpy() for t in tests], axis=0))
    return np.mean(per_base, axis=0)


full_paths = {(b, s): OUTPUT_DIR / f'FULL_{b}_seed{s}__test.csv' for b in FINAL_MODEL_NAMES for s in SEED_SUFFIXES}
missing = [p.name for p in full_paths.values() if not p.is_file()]
if missing:
    print(f'Modèles complets manquants : {len(missing)} (exemple : {missing[0]}). Relancer cette cellule une fois les modèles terminés.')
else:
    per_base = []
    for base in FINAL_MODEL_NAMES:
        frames = [pd.read_csv(full_paths[(base, s)], dtype={'id': str}) for s in SEED_SUFFIXES]
        for f in frames:
            require(f['id'].equals(sample_submission['id']), 'Ordre des ID test incohérent entre modèles complets')
        per_base.append(np.mean([f['accepted'].to_numpy() for f in frames], axis=0))
    full = np.mean(per_base, axis=0)
    fold_avg = fold_average_multiseed_ensemble()
    print(f'Corrélation test : ensemble complet / moyenne des folds l’ensemble multigraines : {np.corrcoef(full, fold_avg)[0, 1]:.5f}')
    for name, probability in [(ENSEMBLE_FULL_MULTISEED, full), ('hybrid_50_50', 0.5 * full + 0.5 * fold_avg)]:
        require(np.isfinite(probability).all() and len(probability) == 45000, f'{name} : probabilités invalides')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
            path, index=False, float_format='%.12f')
        print('Fichier écrit :', path.name)
    if 'curve' in globals() and 'gain_84k_vs_67k' in curve and curve['gain_84k_vs_67k'].notna().sum() == 5:
        gain = curve['gain_84k_vs_67k']
        if gain.mean() >= 0.0005 and (gain > 0).sum() >= 4:
            advice = 'La courbe soutient le réentraînement complet. Choix final documenté : submission_full_Bseeds.csv (score public 0.92301) ; hybrid et l’ensemble multigraines restent des références.'
        elif gain.mean() >= 0.0002:
            advice = 'Gain limité sur la courbe d’apprentissage. Choix final documenté : submission_full_Bseeds.csv (score public 0.92301) ; hybrid et l’ensemble multigraines restent des références.'
        else:
            advice = 'La courbe ne montre pas de gain lié au volume. Choix final documenté : submission_full_Bseeds.csv (score public 0.92301), retenu après comparaison des soumissions ; ce diagnostic seul ne le justifie pas.'
        print('Bilan : ' + advice)
