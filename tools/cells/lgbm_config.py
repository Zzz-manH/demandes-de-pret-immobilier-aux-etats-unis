EXPERIMENT_ID = 'lightgbm'
FOLDS = [1, 2, 3, 4, 5]
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = ARTIFACTS / '05_error_analysis' / EXPERIMENT_ID

CORE_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing']
ENRICHED_FEATURE_BLOCKS = CORE_FEATURE_BLOCKS + ['precision', 'thresholds', 'implied', 'term']   # Les mêmes 72 colonnes que les CatBoost finaux les deux modèles enrichis
LGBM_BLOCKS = ENRICHED_FEATURE_BLOCKS
LGBM_BASE = dict(
    objective='binary', learning_rate=0.03, n_estimators=MAX_ITERATIONS,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, cat_l2=10.0, cat_smooth=20.0, min_data_per_group=50,
    max_cat_threshold=64, random_state=SEED, n_jobs=os.cpu_count() or 2, verbose=-1,
)
LGBM_CONFIGS = {                      # Deux capacités fixées, sans recherche ; moyenne à parts égales dans l’ensemble
    LIGHTGBM_LEAVES31: dict(num_leaves=31, min_child_samples=50),
    LIGHTGBM_LEAVES127: dict(num_leaves=127, min_child_samples=100),
}
