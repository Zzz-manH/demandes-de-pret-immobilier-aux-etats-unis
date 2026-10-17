fixed_config(OUTPUT_DIR, {
    'purpose': 'TabM-mini (PyTorch) on G7 features; inner 15% early stop, no outer refit',
    'seed': SEED, 'nn_config': NN_CONFIG, 'blocks': G7, 'data_hashes': DATA_HASHES,
})
for fold in FOLDS:
    path = OUTPUT_DIR / f'TabMMini__fold{fold}.json'
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf-8'))
        print(f'TabMMini fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
        continue
    result = train_fold(fold)
    print(f'TabMMini fold {fold}: AUC={result["valid_auc"]:.6f}  最佳 epoch={result["best_epoch"]}  '
          f'{result["seconds"]/60:.1f} 分钟', flush=True)
