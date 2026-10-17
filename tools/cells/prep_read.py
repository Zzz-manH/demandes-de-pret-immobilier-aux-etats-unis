SOURCE_FILES = ["train.csv", "test.csv", "sample_submission.csv"]
for name in SOURCE_FILES:
    if not (DATA_DIR / name).is_file():
        raise FileNotFoundError(f"缺少 {DATA_DIR / name}：原始数据应放在项目的 data/ 目录。")

def read_text_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

train_raw = read_text_csv(DATA_DIR / "train.csv")
test_raw = read_text_csv(DATA_DIR / "test.csv")
sample_raw = read_text_csv(DATA_DIR / "sample_submission.csv")
assert {"id", "accepted"}.issubset(train_raw.columns)
assert "id" in test_raw and "accepted" not in test_raw
assert sample_raw.columns.tolist() == ["id", "accepted"], "提交模板字段变化，先核查接口"
feature_columns = train_raw.columns.drop(["id", "accepted"]).tolist()
assert feature_columns == test_raw.columns.drop("id").tolist(), "训练/测试特征或顺序不一致"
assert len(train_raw) > 0 and len(test_raw) > 0
for name, frame in [("train", train_raw), ("test", test_raw), ("sample", sample_raw)]:
    assert frame.columns.is_unique, f"{name}: 重复列名"
    assert frame["id"].str.strip().ne("").all() and frame["id"].is_unique, f"{name}: ID 无效"
    assert frame["id"].eq(frame["id"].str.strip()).all(), f"{name}: ID 有首尾空格"
assert set(train_raw["id"]).isdisjoint(test_raw["id"]), "train/test ID 重叠"
assert set(sample_raw["id"]) == set(test_raw["id"]), "提交模板与测试 ID 不匹配"
assert train_raw["accepted"].isin(["0", "1"]).all(), "accepted 必须为 0 或 1"
y = train_raw["accepted"].astype("int8")
assert set(y.unique()) == {0, 1}, "二分类训练数据必须包含两个类别"
train_labels = pd.DataFrame({"id": train_raw["id"], "accepted": y})
test_ids = test_raw[["id"]].copy()
display(pd.DataFrame({
    "数据": ["train", "test"], "行数": [len(train_raw), len(test_raw)],
    "特征列": [len(feature_columns)] * 2,
}))
display(y.value_counts().sort_index().rename("数量").to_frame().assign(比例=lambda t: t["数量"] / len(y)))
