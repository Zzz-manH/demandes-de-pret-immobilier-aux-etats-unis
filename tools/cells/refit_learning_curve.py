if RUN_LEARNING_CURVE:
    lc_spec = GROUPS[LEARNING_CURVE_MODEL]
    lc_train, _, lc_cats = variant_frames(lc_spec['blocks'])
    for fold in [1, 2, 3, 4, 5]:
        fit_idx = np.flatnonzero(fold_assignment != fold)
        valid_idx = np.flatnonzero(fold_assignment == fold)
        sub_idx, _ = train_test_split(fit_idx, train_size=LEARNING_CURVE_FRACTION, random_state=SEED + fold,
                                      stratify=y.iloc[fit_idx])
        for name, idx in [(LEARNING_CURVE_SUBSET, np.sort(sub_idx)), (LEARNING_CURVE_FULL, fit_idx)]:
            path = OUTPUT_DIR / f'{name}__fold{fold}.json'
            if path.is_file():
                saved = json.loads(path.read_text(encoding='utf-8'))
                print(f'{name} fold {fold}: terminé, AUC={saved["valid_auc"]:.6f}, ignoré')
                continue
            started = time.perf_counter()
            model, best, rounds = early_stop_refit(lc_spec['params'], lc_train, lc_cats, idx, SEED)
            probability = checked_probabilities(model, lc_train.iloc[valid_idx])
            result = {'name': name, 'fold': fold, 'group': LEARNING_CURVE_MODEL, 'train_rows': int(len(idx)),
                      'inner_best_iteration': best, 'refit_rounds': rounds,
                      'valid_auc': roc_auc_score(y.iloc[valid_idx], probability),
                      'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                      'finished_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json(path, result)
            print(f'{name} fold {fold}: {len(idx):,} lignes  AUC={result["valid_auc"]:.6f}  '
                  f'itérations={rounds}  {result["seconds"]/60:.1f} min', flush=True)
            del model
            gc.collect()

lc = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT_DIR.glob(LEARNING_CURVE_RUN_PATTERN)])
if not lc.empty:
    curve = lc.pivot(index='fold', columns='name', values='valid_auc')
    if {LEARNING_CURVE_SUBSET, LEARNING_CURVE_FULL} <= set(curve.columns):
        curve['gain_84k_vs_67k'] = curve[LEARNING_CURVE_FULL] - curve[LEARNING_CURVE_SUBSET]
    reference = {k: json.loads((CATBOOST_DIR / f'{LEARNING_CURVE_MODEL}__fold{k}.json').read_text(encoding='utf-8'))['valid_auc']
                 for k in curve.index}
    curve[REDUCED_MODEL_CV_AUC] = pd.Series(reference)
    display_named(curve.round(6))
    if 'gain_84k_vs_67k' in curve:
        done = curve['gain_84k_vs_67k'].dropna()
        print(f'Données ×1,25 (67k → 84k) : gain moyen d’AUC {done.mean():+.6f}, {int((done > 0).sum())}/{len(done)} folds en progrès.'
              f' Le réentraînement complet (84k → 105k) augmente aussi les données de ×1,25 ; un gain comparable reste une hypothèse.')
