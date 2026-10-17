complete = pl_gain is not None and pl_gain[2] == 5
passed = complete and pl_gain[0] >= 0.0003 and pl_gain[1] >= 4
do_final = passed if RUN_FINAL_PL == 'auto' else bool(RUN_FINAL_PL)
if not do_final:
    print('跳过全量伪标签训练：' + ('验证未完成。' if not complete else
          f'验证未通过（平均 {pl_gain[0]:+.6f}，{pl_gain[1]}/5 折）。最终提交保持 04 的 full_Bseeds。'))
else:
    ratio = float(table['rounds_ratio'].mean())
    full_teacher = mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for b in CAT_BASES for s in SEED_SUFFIXES])
    all_idx = np.arange(len(y))
    for base in CAT_BASES:
        spec = GROUPS[base]
        train, test, cats = variant_frames(spec['blocks'])
        for seed in SEED_SUFFIXES:
            name = f'FULLPL_{base}_seed{seed}'
            path = OUTPUT_DIR / f'{name}.json'
            if path.is_file():
                print(f'{name}: 已完成，跳过')
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
            require(int(model.tree_count_) == rounds, '全量伪标签模型轮数不符')
            probability = checked_probabilities(model, test)
            pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': probability}).to_csv(
                OUTPUT_DIR / f'{name}__test.csv', index=False, float_format='%.12f')
            write_json(path, {'name': name, 'base': base, 'seed': seed, 'rounds': rounds,
                              'rounds_ratio': ratio, 'seconds': time.perf_counter() - started,
                              'gpu_name': GPU_NAME, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
            print(f'{name}: {rounds} 轮  {(time.perf_counter() - started)/60:.1f} 分钟', flush=True)
            del model, X_all
            gc.collect()

    students = np.mean([mean_test([OUTPUT_DIR / f'FULLPL_{b}_seed{s}__test.csv' for s in SEED_SUFFIXES])
                        for b in CAT_BASES], axis=0)
    full_bseeds = np.mean([mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for s in SEED_SUFFIXES])
                           for b in CAT_BASES], axis=0)
    print(f'伪标签全量融合 与 04 的 full_Bseeds 测试预测相关系数：{np.corrcoef(students, full_bseeds)[0, 1]:.5f}')
    for name, probability in [('full_PL', students), ('full_PL_mix', 0.5 * students + 0.5 * full_bseeds)]:
        require(np.isfinite(probability).all() and len(probability) == 45000, f'{name} 概率无效')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
            path, index=False, float_format='%.12f')
        print('写出', path.name)
    print('建议：验证通过时 submission_full_PL.csv 作主选，04 的 submission_full_Bseeds.csv 作备选。')
