if RUN_LEARNING_CURVE:
    lc_spec = GROUPS[LC_GROUP]
    lc_train, _, lc_cats = variant_frames(lc_spec['blocks'])
    for fold in [1, 2, 3, 4, 5]:
        fit_idx = np.flatnonzero(fold_assignment != fold)
        valid_idx = np.flatnonzero(fold_assignment == fold)
        sub_idx, _ = train_test_split(fit_idx, train_size=LC_FRACTION, random_state=SEED + fold,
                                      stratify=y.iloc[fit_idx])
        for name, idx in [('LC_sub80', np.sort(sub_idx)), ('LC_full', fit_idx)]:
            path = OUTPUT_DIR / f'{name}__fold{fold}.json'
            if path.is_file():
                saved = json.loads(path.read_text(encoding='utf-8'))
                print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
                continue
            started = time.perf_counter()
            model, best, rounds = early_stop_refit(lc_spec['params'], lc_train, lc_cats, idx, SEED)
            probability = checked_probabilities(model, lc_train.iloc[valid_idx])
            result = {'name': name, 'fold': fold, 'group': LC_GROUP, 'train_rows': int(len(idx)),
                      'inner_best_iteration': best, 'refit_rounds': rounds,
                      'valid_auc': roc_auc_score(y.iloc[valid_idx], probability),
                      'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                      'finished_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json(path, result)
            print(f'{name} fold {fold}: {len(idx):,} 行  AUC={result["valid_auc"]:.6f}  '
                  f'轮数={rounds}  {result["seconds"]/60:.1f} 分钟', flush=True)
            del model
            gc.collect()

lc = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT_DIR.glob('LC_*__fold?.json')])
if not lc.empty:
    curve = lc.pivot(index='fold', columns='name', values='valid_auc')
    if {'LC_sub80', 'LC_full'} <= set(curve.columns):
        curve['gain_84k_vs_67k'] = curve['LC_full'] - curve['LC_sub80']
    reference = {k: json.loads((CATBOOST_DIR / f'{LC_GROUP}__fold{k}.json').read_text(encoding='utf-8'))['valid_auc']
                 for k in curve.index}
    curve['M3_in_13'] = pd.Series(reference)
    display(curve.round(6))
    if 'gain_84k_vs_67k' in curve:
        done = curve['gain_84k_vs_67k'].dropna()
        print(f'训练数据 ×1.25（67k → 84k）：平均 AUC {done.mean():+.6f}，{int((done > 0).sum())}/{len(done)} 折为正。'
              f' 全量重训（84k → 105k）是同样的 ×1.25，预期收益与此同量级。')
