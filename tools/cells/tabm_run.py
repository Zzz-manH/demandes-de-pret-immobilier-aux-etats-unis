fixed_config(OUTPUT_DIR, {
    'purpose': PURPOSE_TABM,
    'seed': SEED, 'nn_config': NN_CONFIG, 'blocks': ENRICHED_FEATURE_BLOCKS, 'data_hashes': DATA_HASHES,
})
for fold in FOLDS:
    path = OUTPUT_DIR / f'TabMMini__fold{fold}.json'
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf-8'))
        print(f'TabMMini fold {fold}: terminé, AUC={saved["valid_auc"]:.6f}, ignoré')
        continue
    result = train_fold(fold)
    print(f'TabMMini fold {fold}: AUC={result["valid_auc"]:.6f}  meilleure epoch={result["best_epoch"]}  '
          f'{result["seconds"]/60:.1f} min', flush=True)
