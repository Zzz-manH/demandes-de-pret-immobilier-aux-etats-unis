EXPERIMENT_ID = 'full_refit'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = ARTIFACTS / '04_final_model' / EXPERIMENT_ID

GPU_BASE = dict(
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    allow_writing_files=False, verbose=False,
)
ALL_BLOCKS = ['dti', 'ltv', 'income', 'missing', 'freq']
NO_FREQ = ['dti', 'ltv', 'income', 'missing']
FAST = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3,
            bootstrap_type='Bernoulli', subsample=0.8, early_stop=False, iterations=1600)
C7 = dict(depth=8, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
C11 = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
ROUND2 = ['precision', 'thresholds', 'implied', 'term']
G7 = NO_FREQ + ROUND2                      # 02 特征工程中最好的特征集（72 列）
LEAN = NO_FREQ + ['precision', 'thresholds']  # 第二轮只保留单独有效的两块（68 列）
GROUPS = {
    'M0_C11_all': dict(params=C11, blocks=ALL_BLOCKS),   # 第一轮全部特征（公开榜 0.92073 的配置），同轮对照
    'M1_C11_G7': dict(params=C11, blocks=G7),
    'M2_C7_G7': dict(params=C7, blocks=G7),
    'M3_C11_lean': dict(params=C11, blocks=LEAN),
}
BASELINE = 'M0_C11_all'
CATBOOST_DIR = ARTIFACTS / '04_final_model' / 'cv_models'   # 第 1 节的五折模型结果
CAT_BASES = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']
SEED_SUFFIXES = {42: '', 2027: '_s2027', 31415: '_s31415'}
LC_GROUP = 'M3_C11_lean'       # 学习曲线只用五折最好的单组
LC_FRACTION = 0.8              # 外层训练部分的 80%（67,200 行）对比 100%（84,000 行）
FULL_ROUNDS_SCALE = 105000 / 84000   # 全量轮数 = 该组五折重训轮数均值 × 数据量之比（与 b/0.85 同一规则）
RUN_LEARNING_CURVE = TRAIN['learning_curve']
RUN_FULL_REFIT = TRAIN['full_refit']
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(GROUPS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知组：{sorted(unknown)}')
