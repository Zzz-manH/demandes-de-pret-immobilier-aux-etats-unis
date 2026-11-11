EXPERIMENT_ID = 'catboost_tuning'
FOLDS = [1, 2, 3, 4, 5]     # Deuxième série : confirmation sur cinq folds ; [1] pour un tri sur le premier fold
RUN_ONLY = [TUNING_DEPTH8_FIXED1600, TUNING_DEPTH8_BERNOULLI80,
            TUNING_DEPTH8_L2_30_BERNOULLI80, TUNING_DEPTH6_BERNOULLI80]  # None sélectionne tous les candidats
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / EXPERIMENT_ID

GPU_BASE = dict(
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    allow_writing_files=False, verbose=False,
)
# early_stop=False : nombre d’itérations fixe ; ordre croissant de durée estimée
CANDIDATES = {
    TUNING_DEPTH8_FIXED1600:   dict(depth=8, learning_rate=0.05, l2_leaf_reg=3, early_stop=False, iterations=1600),
    TUNING_DEPTH8_LR005_L2_3: dict(depth=8, learning_rate=0.05, l2_leaf_reg=3),
    TUNING_DEPTH6_LR003_L2_10:  dict(depth=6, learning_rate=0.03, l2_leaf_reg=10),
    TUNING_DEPTH8_LR003_L2_10:  dict(depth=8, learning_rate=0.03, l2_leaf_reg=10),
    TUNING_DEPTH8_LR003_L2_3:   dict(depth=8, learning_rate=0.03, l2_leaf_reg=3),
    TUNING_DEPTH8_LR003_L2_30:  dict(depth=8, learning_rate=0.03, l2_leaf_reg=30),
    TUNING_DEPTH8_BERNOULLI80: dict(depth=8, learning_rate=0.03, l2_leaf_reg=10,
                                   bootstrap_type='Bernoulli', subsample=0.8),
    TUNING_DEPTH8_WITH_DTI: dict(depth=8, learning_rate=0.03, l2_leaf_reg=10, dti=True),
    TUNING_DEPTH10_LR003_L2_10: dict(depth=10, learning_rate=0.03, l2_leaf_reg=10),
    TUNING_DEPTH8_LR002_L2_10:  dict(depth=8, learning_rate=0.02, l2_leaf_reg=10),
    # Deuxième série : combiner les trois meilleurs du fold 1 (sous-échantillonnage, depth 6, L2=30)
    TUNING_DEPTH8_L2_30_BERNOULLI80: dict(depth=8, learning_rate=0.03, l2_leaf_reg=30,
                                    bootstrap_type='Bernoulli', subsample=0.8),
    TUNING_DEPTH6_BERNOULLI80: dict(depth=6, learning_rate=0.03, l2_leaf_reg=10,
                                    bootstrap_type='Bernoulli', subsample=0.8),
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(CANDIDATES)
    if unknown:
        raise ValueError(f'Candidats inconnus dans RUN_ONLY : {sorted(unknown)}')
