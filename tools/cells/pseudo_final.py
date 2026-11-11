complete = pl_gain is not None and pl_gain[2] == 5
passed = complete and pl_gain[0] >= 0.0003 and pl_gain[1] >= 4
do_final = passed if RUN_FINAL_PL == 'auto' else bool(RUN_FINAL_PL)
if not do_final:
    print('Entraînement complet avec pseudo-labels ignoré : ' + ('validation incomplète.' if not complete else
          f'critère de validation non atteint (moyenne {pl_gain[0]:+.6f}, {pl_gain[1]}/5 folds). La soumission finale reste l’ensemble complet multigraines du notebook 04.'))
else:
    ratio = float(table['rounds_ratio'].mean())
    full_teacher = mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for b in FINAL_MODEL_NAMES for s in SEED_SUFFIXES])
    all_idx = np.arange(len(y))
    for base in FINAL_MODEL_NAMES:
        spec = GROUPS[base]
        train, test, cats = variant_frames(spec['blocks'])
        for seed in SEED_SUFFIXES:
            name = f'FULLPL_{base}_seed{seed}'
            path = OUTPUT_DIR / f'{name}.json'
            if path.is_file():
                print(f'{name}: terminé, ignoré')
                continue
            full_info = json.loads((FULL_DIR / f'FULL_{base}_seed{seed}.json').read_text(encoding='utf-8'))
            rounds = int(round(full_info['rounds'] * ratio))
            started = time.perf_counter()
            X_all = pd.concat([train, test], ignore_index=True)
            y_all = np.concatenate([y.to_numpy(dtype='float64'), full_teacher])
            w_all = np.concatenate([np.ones(len(train)), np.full(len(test), PL_WEIGHT)])
            model = catboost.CatBoostClassifier(**model_params(spec['params'], rounds, seed, loss='CrossEntropy'),
                                                use_best_model=False)
            model.fit(X_all, y_all, sample_weight=w_all, cat_features=cats)
            require(int(model.tree_count_) == rounds, 'Nombre d’itérations incorrect pour le modèle complet à pseudo-labels')
            probability = checked_probabilities(model, test)
            pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': probability}).to_csv(
                OUTPUT_DIR / f'{name}__test.csv', index=False, float_format='%.12f')
            write_json(path, {'name': name, 'base': base, 'seed': seed, 'rounds': rounds,
                              'rounds_ratio': ratio, 'seconds': time.perf_counter() - started,
                              'gpu_name': GPU_NAME, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
            print(f'{name}: {rounds} itérations  {(time.perf_counter() - started)/60:.1f} min', flush=True)
            del model, X_all
            gc.collect()

    students = np.mean([mean_test([OUTPUT_DIR / f'FULLPL_{b}_seed{s}__test.csv' for s in SEED_SUFFIXES])
                        for b in FINAL_MODEL_NAMES], axis=0)
    full_multiseed_ensemble = np.mean([mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for s in SEED_SUFFIXES])
                           for b in FINAL_MODEL_NAMES], axis=0)
    print(f'Corrélation des prédictions test : ensemble complet à pseudo-labels / l’ensemble complet multigraines de 04 : {np.corrcoef(students, full_multiseed_ensemble)[0, 1]:.5f}')
    for name, probability in [('full_PL', students), ('full_PL_mix', 0.5 * students + 0.5 * full_multiseed_ensemble)]:
        require(np.isfinite(probability).all() and len(probability) == 45000, f'{name} : probabilités invalides')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
            path, index=False, float_format='%.12f')
        print('Fichier écrit :', path.name)
    print('Si la validation est concluante, submission_full_PL.csv devient un candidat ; la référence finale documentée reste submission_full_Bseeds.csv de 04.')
