EXPERIMENT_ID = 'ablation_round3'
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
NO_FREQ = ['dti', 'ltv', 'income', 'missing']
G7 = NO_FREQ + ['precision', 'thresholds', 'implied', 'term']   # 第二轮的最佳特征集（72 列）
ROUND3 = ['purpose', 'ltv_consistency', 'risk_layers', 'tract_income']
VARIANTS = {
    'H0_G7': G7,
    'H1_purpose': G7 + ['purpose'],
    'H2_ltv_consistency': G7 + ['ltv_consistency'],
    'H3_risk_layers': G7 + ['risk_layers'],
    'H4_tract_income': G7 + ['tract_income'],
    'H5_round3_all': G7 + ROUND3,
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知变体：{sorted(unknown)}')
