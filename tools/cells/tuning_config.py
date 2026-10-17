EXPERIMENT_ID = 'catboost_tuning'
FOLDS = [1, 2, 3, 4, 5]     # 第二轮：五折复核；只做第 1 折筛选时改回 [1]
RUN_ONLY = ['B0_R2_fixed1600', 'C7_d8_lr03_l2_10_sub80',
            'C10_d8_lr03_l2_30_sub80', 'C11_d6_lr03_l2_10_sub80']  # None 表示全部候选
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / EXPERIMENT_ID

GPU_BASE = dict(
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    allow_writing_files=False, verbose=False,
)
# early_stop=False 表示固定轮数、不早停；按预计耗时从小到大排列
CANDIDATES = {
    'B0_R2_fixed1600':   dict(depth=8, learning_rate=0.05, l2_leaf_reg=3, early_stop=False, iterations=1600),
    'C1_d8_lr05_l2_3_es': dict(depth=8, learning_rate=0.05, l2_leaf_reg=3),
    'C6_d6_lr03_l2_10':  dict(depth=6, learning_rate=0.03, l2_leaf_reg=10),
    'C3_d8_lr03_l2_10':  dict(depth=8, learning_rate=0.03, l2_leaf_reg=10),
    'C2_d8_lr03_l2_3':   dict(depth=8, learning_rate=0.03, l2_leaf_reg=3),
    'C4_d8_lr03_l2_30':  dict(depth=8, learning_rate=0.03, l2_leaf_reg=30),
    'C7_d8_lr03_l2_10_sub80': dict(depth=8, learning_rate=0.03, l2_leaf_reg=10,
                                   bootstrap_type='Bernoulli', subsample=0.8),
    'C8_d8_lr03_l2_10_DTI': dict(depth=8, learning_rate=0.03, l2_leaf_reg=10, dti=True),
    'C5_d10_lr03_l2_10': dict(depth=10, learning_rate=0.03, l2_leaf_reg=10),
    'C9_d8_lr02_l2_10':  dict(depth=8, learning_rate=0.02, l2_leaf_reg=10),
    # 第二轮：把第 1 折前三名（抽样、depth 6、L2=30）组合起来
    'C10_d8_lr03_l2_30_sub80': dict(depth=8, learning_rate=0.03, l2_leaf_reg=30,
                                    bootstrap_type='Bernoulli', subsample=0.8),
    'C11_d6_lr03_l2_10_sub80': dict(depth=6, learning_rate=0.03, l2_leaf_reg=10,
                                    bootstrap_type='Bernoulli', subsample=0.8),
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(CANDIDATES)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知候选：{sorted(unknown)}')
