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
INITIAL_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing', 'freq']
CORE_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing']
FAST_SCREEN_PARAMS = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3,
            bootstrap_type='Bernoulli', subsample=0.8, early_stop=False, iterations=1600)
CATBOOST_DEPTH8_PARAMS = dict(depth=8, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
CATBOOST_DEPTH6_PARAMS = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
LTV_DETAIL_BLOCKS = ['precision', 'thresholds', 'implied', 'term']
ENRICHED_FEATURE_BLOCKS = CORE_FEATURE_BLOCKS + LTV_DETAIL_BLOCKS                      # Meilleur jeu de variables du notebook 02 (72 colonnes)
REDUCED_FEATURE_BLOCKS = CORE_FEATURE_BLOCKS + ['precision', 'thresholds']  # Deux blocs de la deuxième série utiles isolément (68 colonnes)
GROUPS = {
    MODEL_DEPTH6_INITIAL: dict(params=CATBOOST_DEPTH6_PARAMS, blocks=INITIAL_FEATURE_BLOCKS),   # Toutes les variables de la première série ; référence comparable (score public 0.92073)
    MODEL_DEPTH6_ENRICHED: dict(params=CATBOOST_DEPTH6_PARAMS, blocks=ENRICHED_FEATURE_BLOCKS),
    MODEL_DEPTH8_ENRICHED: dict(params=CATBOOST_DEPTH8_PARAMS, blocks=ENRICHED_FEATURE_BLOCKS),
    MODEL_DEPTH6_REDUCED: dict(params=CATBOOST_DEPTH6_PARAMS, blocks=REDUCED_FEATURE_BLOCKS),
}
BASELINE = MODEL_DEPTH6_INITIAL
CATBOOST_DIR = ARTIFACTS / '04_final_model' / 'cv_models'   # Résultats des cinq folds de la section 1
FINAL_MODEL_NAMES = [MODEL_DEPTH6_ENRICHED, MODEL_DEPTH8_ENRICHED, MODEL_DEPTH6_REDUCED]
SEED_SUFFIXES = {42: '', 2027: '_s2027', 31415: '_s31415'}
LEARNING_CURVE_MODEL = MODEL_DEPTH6_REDUCED       # Courbe d’apprentissage : meilleur groupe individuel sur cinq folds
LEARNING_CURVE_FRACTION = 0.8              # 80 % du train externe (67 200 lignes) contre 100 % (84 000 lignes)
FULL_ROUNDS_SCALE = 105000 / 84000   # Itérations complètes = moyenne des cinq réentraînements × ratio des effectifs (même règle que b/0.85)
RUN_LEARNING_CURVE = TRAIN['learning_curve']
RUN_FULL_REFIT = TRAIN['full_refit']
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(GROUPS)
    if unknown:
        raise ValueError(f'Groupes inconnus dans RUN_ONLY : {sorted(unknown)}')
