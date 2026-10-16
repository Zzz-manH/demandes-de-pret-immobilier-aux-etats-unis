def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = PREPARED_DIR / 'manifest.json'
schema_path = PREPARED_DIR / 'schema.json'
if not manifest_path.is_file() or not schema_path.is_file():
    raise FileNotFoundError(
        f'缺少 {manifest_path} 或 {schema_path}；请先运行 01_hmda_data_preparation.ipynb'
    )
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
schema = json.loads(schema_path.read_text(encoding='utf-8'))
require(manifest['schema_version'] == schema['schema_version'] == '1.0', '数据契约版本不匹配')
require(manifest['seed'] == SEED and manifest['n_splits'] == 5, '五折配置不匹配')
require(set(manifest['source_files']) == {'train.csv', 'test.csv', 'sample_submission.csv'},
        '原始文件清单不完整')
expected_prepared = {
    'train_features.csv', 'test_features.csv', 'train_labels.csv',
    'test_ids.csv', 'folds.csv', 'audit.csv', 'schema.json',
}
require(set(manifest['output_files']) == expected_prepared, '准备文件清单不完整')
for name, info in manifest['source_files'].items():
    path = PROJECT_DIR / name
    require(path.is_file(), f'缺少原始数据：{path}')
    require(sha256_file(path) == info['sha256'], f'原始文件已变化：{name}；请先重跑 01')
for name, info in manifest['output_files'].items():
    path = PREPARED_DIR / name
    require(path.is_file(), f'缺少准备文件：{path}')
    require(sha256_file(path) == info['sha256'], f'准备文件哈希不匹配：{name}；请先重跑 01')

FEATURE_COLUMNS = schema['feature_columns']
NUMERIC_COLUMNS = schema['numeric_columns']
CATEGORICAL_COLUMNS = schema['categorical_columns']
require(len(FEATURE_COLUMNS) == 56 and len(NUMERIC_COLUMNS) == 13
        and len(CATEGORICAL_COLUMNS) == 43, '预期为 56 列：13 数值、43 类别')
require(set(FEATURE_COLUMNS) == set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS),
        '数值/类别列分组不完整')
require(not {'id', 'accepted'} & set(FEATURE_COLUMNS), 'ID 或标签不能作为特征')


def read_features(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(frame.columns.tolist() == FEATURE_COLUMNS, f'特征列或顺序不匹配：{path.name}')
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column].replace('', np.nan), errors='raise',
        ).astype('float64')
    require(not np.isinf(frame[NUMERIC_COLUMNS].to_numpy()).any(), '数值列含无穷大')
    require(not frame[CATEGORICAL_COLUMNS].eq('').any().any(), '类别列含空字符串')
    return frame


X_base = read_features(PREPARED_DIR / 'train_features.csv')
X_test_base = read_features(PREPARED_DIR / 'test_features.csv')
labels = pd.read_csv(PREPARED_DIR / 'train_labels.csv', dtype=str, keep_default_na=False)
test_id_frame = pd.read_csv(PREPARED_DIR / 'test_ids.csv', dtype=str, keep_default_na=False)
fold_frame = pd.read_csv(PREPARED_DIR / 'folds.csv', dtype=str, keep_default_na=False)
sample_submission = pd.read_csv(PROJECT_DIR / 'sample_submission.csv', dtype=str, keep_default_na=False)

require(labels.columns.tolist() == ['id', 'accepted'], '训练标签列不匹配')
require(test_id_frame.columns.tolist() == ['id'], '测试 ID 列不匹配')
require(fold_frame.columns.tolist() == ['id', 'fold'], '折号文件列不匹配')
require(sample_submission.columns.tolist() == ['id', 'accepted'], '提交模板列不匹配')
require(len(X_base) == len(labels) == len(fold_frame) == 105000, '训练行数不匹配')
require(len(X_test_base) == len(test_id_frame) == len(sample_submission) == 45000,
        '测试行数不匹配')

train_ids = labels['id']
test_ids = test_id_frame['id']
require(train_ids.is_unique and test_ids.is_unique, 'ID 重复')
require(train_ids.ne('').all() and test_ids.ne('').all(), 'ID 为空')
require(train_ids.equals(fold_frame['id']), '折号 ID 与训练行错位')
require(test_ids.equals(sample_submission['id']), '测试 ID 与提交模板顺序不符')
require(set(train_ids).isdisjoint(test_ids), '训练与测试 ID 重叠')
require(set(labels['accepted']) == {'0', '1'}, 'accepted 需同时包含 0 和 1')
raw_train_identity = pd.read_csv(
    PROJECT_DIR / 'train.csv', usecols=['id', 'accepted'],
    dtype=str, keep_default_na=False,
)
raw_test_identity = pd.read_csv(
    PROJECT_DIR / 'test.csv', usecols=['id'], dtype=str, keep_default_na=False,
)
require(train_ids.equals(raw_train_identity['id'])
        and labels['accepted'].equals(raw_train_identity['accepted']),
        '准备标签与原始 train.csv 的 ID/标签不一致')
require(test_ids.equals(raw_test_identity['id']),
        '准备测试 ID 与原始 test.csv 不一致')
y = labels['accepted'].astype('int8')
require(set(fold_frame['fold']) == {'1', '2', '3', '4', '5'}, '折号应为 1..5')
fold_assignment = fold_frame['fold'].astype('int8').to_numpy()

expected_folds = np.zeros(len(X_base), dtype=np.int8)
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (_, valid_index) in enumerate(splitter.split(X_base, y), start=1):
    expected_folds[valid_index] = fold
require(np.array_equal(fold_assignment, expected_folds), '折号并非固定 seed=42 五折')
DATA_HASHES = {
    'manifest.json': sha256_file(manifest_path),
    **{name: info['sha256'] for name, info in manifest['output_files'].items()},
}
print(f'数据核对通过：训练 {len(X_base):,}，测试 {len(X_test_base):,}，56 个原始特征。')
display(pd.crosstab(fold_assignment, y, rownames=['fold'], colnames=['accepted']))