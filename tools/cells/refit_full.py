def group_rounds(group):
    rounds = [json.loads((CATBOOST_DIR / f'{group}__fold{k}.json').read_text(encoding='utf-8'))['refit_rounds']
              for k in range(1, 6)]
    return float(np.mean(rounds))


if RUN_FULL_REFIT:
    for base in FINAL_MODEL_NAMES:
        spec = GROUPS[base]
        train, test, cats = variant_frames(spec['blocks'])
        for seed, suffix in SEED_SUFFIXES.items():
            name = f'FULL_{base}_seed{seed}'
            path = OUTPUT_DIR / f'{name}.json'
            if path.is_file():
                print(f'{name}: terminé, ignoré')
                continue
            rounds = int(round(group_rounds(base + suffix) * FULL_ROUNDS_SCALE))
            started = time.perf_counter()
            model = catboost.CatBoostClassifier(**model_params(spec['params'], rounds, seed),
                                                use_best_model=False)
            model.fit(train, y, cat_features=cats)
            require(int(model.tree_count_) == rounds, 'Nombre d’itérations incorrect pour le modèle complet')
            probability = checked_probabilities(model, test)
            pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': probability}).to_csv(
                OUTPUT_DIR / f'{name}__test.csv', index=False, float_format='%.12f')
            write_json(path, {'name': name, 'base': base, 'seed': seed, 'train_rows': len(train),
                              'rounds': rounds, 'seconds': time.perf_counter() - started,
                              'gpu_name': GPU_NAME,
                              'finished_at_utc': datetime.now(timezone.utc).isoformat()})
            print(f'{name}: {rounds} itérations  {(time.perf_counter() - started)/60:.1f} min', flush=True)
            del model
            gc.collect()
