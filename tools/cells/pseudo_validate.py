pl_spec = GROUPS[PL_GROUP]
pl_train, pl_test, pl_cats = variant_frames(pl_spec['blocks'])
for fold in [1, 2, 3, 4, 5]:
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    pseudo = teacher_for_fold(fold)
    for name in ['A_base', 'A_pseudo']:
        path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        started = time.perf_counter()
        if name == 'A_base':
            model, best, rounds = early_stop_refit(pl_spec['params'], pl_train, pl_cats, fit_idx, SEED)
        else:
            model, best, rounds = pseudo_label_refit(pl_spec['params'], pl_train, pl_test, pl_cats,
                                                     fit_idx, pseudo, SEED)
        probability = checked_probabilities(model, pl_train.iloc[valid_idx])
        pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': y.iloc[valid_idx].to_numpy(),
                      'prob': probability}).to_csv(OUTPUT_DIR / f'{name}__fold{fold}__valid.csv',
                                                   index=False, float_format='%.12f')
        result = {'name': name, 'fold': fold, 'group': PL_GROUP, 'pl_weight': PL_WEIGHT,
                  'inner_best_iteration': best, 'refit_rounds': rounds,
                  'valid_auc': roc_auc_score(y.iloc[valid_idx], probability),
                  'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                  'finished_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(path, result)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  轮数={rounds}  '
              f'{result["seconds"]/60:.1f} 分钟', flush=True)
        del model
        gc.collect()

res = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT_DIR.glob('A_*__fold?.json')])
table = res.pivot(index='fold', columns='name', values='valid_auc')
pl_gain = None
if {'A_base', 'A_pseudo'} <= set(table.columns):
    table['gain_pseudo'] = table['A_pseudo'] - table['A_base']
    blend_rows = {}
    for fold in table.index:
        if not (OUTPUT_DIR / f'A_pseudo__fold{fold}__valid.csv').is_file():
            continue
        student = pd.read_csv(OUTPUT_DIR / f'A_pseudo__fold{fold}__valid.csv')
        merged = teacher_valid(fold).merge(student, on='id', validate='one_to_one')
        auc_teacher = roc_auc_score(merged['accepted'], merged['teacher'])
        auc_mix = roc_auc_score(merged['accepted'], 0.5 * merged['teacher'] + 0.5 * merged['prob'])
        blend_rows[fold] = {'teacher_9models': auc_teacher, 'mix_gain_vs_teacher': auc_mix - auc_teacher}
    table = table.join(pd.DataFrame(blend_rows).T)
    rounds = res.pivot(index='fold', columns='name', values='refit_rounds')
    table['rounds_ratio'] = rounds['A_pseudo'] / rounds['A_base']
    done = table['gain_pseudo'].dropna()
    pl_gain = (done.mean(), int((done > 0).sum()), len(done))
display(table.round(6))
if pl_gain:
    print(f'伪标签（单模型对单模型）：平均 {pl_gain[0]:+.6f}，{pl_gain[1]}/{pl_gain[2]} 折为正。'
          f' mix_gain_vs_teacher 表示学生与 9 模型老师各半后相对老师本身的变化（参考）。')
