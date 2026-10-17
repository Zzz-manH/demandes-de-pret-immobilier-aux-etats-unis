EXPERIMENT_ID = 'training_modes'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None                 # 例如 ['V0_base', 'V1_ordered']；None 表示全部
INNER_STOP_FRACTION = 0.15
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / EXPERIMENT_ID

NO_FREQ = ['dti', 'ltv', 'income', 'missing']
LEAN = NO_FREQ + ['precision', 'thresholds']        # M3 的 68 列特征
DEMO_DUPLICATES = [                                 # 与 derived_* 重复的人口统计原始编码与「观察」标记
    'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
    'applicant_race-1', 'co-applicant_race-1',
    'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed',
    'applicant_race_observed', 'co-applicant_race_observed',
    'applicant_sex_observed', 'co-applicant_sex_observed',
]
SCREEN_BASE = dict(              # 接近 M3（C11）但学习率加倍以加快筛选；外层验证只评分
    depth=6, learning_rate=0.06, l2_leaf_reg=10,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', eval_metric='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    iterations=6000, od_type='Iter', od_wait=200, use_best_model=True,
    allow_writing_files=False, verbose=False,
)
VARIANTS = {
    'V0_base': {},
    'V1_ordered': {'boosting_type': 'Ordered'},            # 超过 5 万行时 GPU 默认 Plain，从未试过
    'V2_newtonL2': {'score_function': 'NewtonL2'},         # 默认 Cosine
    'V3_depthwise': {'grow_policy': 'Depthwise', 'depth': 7, 'min_data_in_leaf': 50},
    'V4_lossguide': {'grow_policy': 'Lossguide', 'depth': 10, 'max_leaves': 48,
                     'min_data_in_leaf': 50, 'score_function': 'NewtonL2'},
    'V5_poisson': {'bootstrap_type': 'Poisson', 'subsample': 0.8},   # 仅 GPU 支持
    'V6_sub066': {'subsample': 0.66},                      # 更强的行抽样
    'V7_border254': {'border_count': 254},                 # GPU 默认 128 个数值分箱
    'V8_ctr1': {'max_ctr_complexity': 1},                  # 不做类别组合，降低方差
    'V9_drop_demo': {'drop': DEMO_DUPLICATES},             # 删去 10 个重复的人口统计列
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知变体：{sorted(unknown)}')
