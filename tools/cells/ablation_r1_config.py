EXPERIMENT_ID = 'ablation_round1'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None             # 例如 ['F0_base', 'F2_ltv']；None 表示全部变体
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / EXPERIMENT_ID

FAST_PARAMS = dict(
    iterations=1600, depth=8, learning_rate=0.05, l2_leaf_reg=3,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    use_best_model=False, allow_writing_files=False, verbose=False,
)
VARIANTS = {
    'F0_base': [],
    'F1_dti': ['dti'],
    'F2_ltv': ['ltv'],
    'F3_income': ['income'],
    'F4_missing': ['missing'],
    'F5_freq': ['freq'],
    'F6_all': ['dti', 'ltv', 'income', 'missing', 'freq'],
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知变体：{sorted(unknown)}')
