EXPERIMENT_ID = 'ablation_round2'
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
ROUND1 = ['dti', 'ltv', 'income', 'missing', 'freq']     # 第一轮全部特征（F6_all）
NO_FREQ = ['dti', 'ltv', 'income', 'missing']
VARIANTS = {
    'G0_all': ROUND1,
    'G1_precision': ROUND1 + ['precision'],
    'G2_thresholds': ROUND1 + ['thresholds'],
    'G3_implied': ROUND1 + ['implied'],
    'G4_term': ROUND1 + ['term'],
    'G5_round2_all': ROUND1 + ['precision', 'thresholds', 'implied', 'term'],
    # 调参后去掉地区频次的五折与公开榜都略好：检验第二轮特征加在无频次基础上
    'G6_nofreq': NO_FREQ,
    'G7_nofreq_round2_all': NO_FREQ + ['precision', 'thresholds', 'implied', 'term'],
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知变体：{sorted(unknown)}')
