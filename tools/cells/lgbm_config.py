EXPERIMENT_ID = 'lightgbm'
FOLDS = [1, 2, 3, 4, 5]
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = ARTIFACTS / '05_error_analysis' / EXPERIMENT_ID

NO_FREQ = ['dti', 'ltv', 'income', 'missing']
G7 = NO_FREQ + ['precision', 'thresholds', 'implied', 'term']   # 与最终 CatBoost M1/M2 相同的 72 列
LGBM_BLOCKS = G7
LGBM_BASE = dict(
    objective='binary', learning_rate=0.03, n_estimators=MAX_ITERATIONS,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, cat_l2=10.0, cat_smooth=20.0, min_data_per_group=50,
    max_cat_threshold=64, random_state=SEED, n_jobs=os.cpu_count() or 2, verbose=-1,
)
LGBM_CONFIGS = {                      # 不搜索，只取两个容量；融合时两者等权平均
    'L1_lgbm_leaves31': dict(num_leaves=31, min_child_samples=50),
    'L2_lgbm_leaves127': dict(num_leaves=127, min_child_samples=100),
}
