def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if REBUILD_PREPARED or not (PREPARED_DIR / "manifest.json").is_file():
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "train_features.csv": X_train, "test_features.csv": X_test,
        "train_labels.csv": train_labels, "test_ids.csv": test_ids,
        "folds.csv": folds, "audit.csv": audit_table,
    }
    for name, frame in output_frames.items():
        destination = PREPARED_DIR / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.to_csv(temporary, index=False, na_rep="")
        temporary.replace(destination)
    (PREPARED_DIR / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(DATA_DIR),
        "source_files": {name: {"sha256": sha256_file(DATA_DIR / name), "bytes": (DATA_DIR / name).stat().st_size} for name in SOURCE_FILES},
        "output_files": {name: {"sha256": sha256_file(PREPARED_DIR / name), "bytes": (PREPARED_DIR / name).stat().st_size} for name in [*output_frames, "schema.json"]},
        "n_rows": {"train": len(X_train), "test": len(X_test)},
        "n_features": len(feature_columns),
        "seed": RANDOM_STATE, "n_splits": N_SPLITS,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "preprocessing": "deterministic text/type normalization only; no fitted estimator or population-derived threshold",
    }
    # manifest 最后写入；若中途失败，旧 manifest 与文件哈希不匹配，模型入口会停止。
    manifest_temp = PREPARED_DIR / "manifest.json.tmp"
    manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_temp.replace(PREPARED_DIR / "manifest.json")
    print(f"已保存 {len(output_frames) + 2} 个准备文件。")
else:
    manifest = json.loads((PREPARED_DIR / "manifest.json").read_text(encoding="utf-8"))
    print("artifacts/01_data_exploration 已存在，未重写；下一格用本次重新整理的结果逐项核对已保存的文件。")
