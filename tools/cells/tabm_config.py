EXPERIMENT_ID = 'tabm'
FOLDS = [1]                 # 先只跑第 1 折试水；值得继续再改为 [1, 2, 3, 4, 5]
OUTPUT_DIR = ARTIFACTS / '05_error_analysis' / EXPERIMENT_ID
CATBOOST_DIR = ARTIFACTS / '04_final_model' / 'cv_models'   # 04 的五折模型结果

NO_FREQ = ['dti', 'ltv', 'income', 'missing']
G7 = NO_FREQ + ['precision', 'thresholds', 'implied', 'term']   # 与最终 CatBoost 相同的 72 列
NN_CONFIG = dict(
    k=32,                   # TabM 的「集成成员」数：32 个共享大部分权重的 MLP
    width=512, depth=3, dropout=0.1,
    cat_dim=8,              # 每个类别列的嵌入维度
    ple_bins=16,            # 数值列分段线性编码的分段数
    rare_min_count=5,       # 外层训练部分出现少于 5 次的类别并入「未知」
    lr=1e-3, weight_decay=1e-4, batch_size=512,
    max_epochs=60, patience=8,
    inner_stop_fraction=0.15,
)
BLEND_WEIGHTS = [0.1, 0.2, 0.3]
