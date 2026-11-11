EXPERIMENT_ID = 'ablation_round2'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None             # Exemple : [ABLATION_INITIAL_REFERENCE, ABLATION_ADD_LTV_PRECISION] ; None sélectionne toutes les variantes
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / EXPERIMENT_ID

FAST_PARAMS = dict(
    iterations=1600, depth=8, learning_rate=0.05, l2_leaf_reg=3,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    use_best_model=False, allow_writing_files=False, verbose=False,
)
INITIAL_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing', 'freq']     # Toutes les variables de la première série (Toutes les variables initiales)
CORE_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing']
VARIANTS = {
    ABLATION_INITIAL_REFERENCE: INITIAL_FEATURE_BLOCKS,
    ABLATION_ADD_LTV_PRECISION: INITIAL_FEATURE_BLOCKS + ['precision'],
    ABLATION_ADD_LTV_THRESHOLDS: INITIAL_FEATURE_BLOCKS + ['thresholds'],
    ABLATION_ADD_IMPLIED_VALUE: INITIAL_FEATURE_BLOCKS + ['implied'],
    ABLATION_ADD_TERM: INITIAL_FEATURE_BLOCKS + ['term'],
    ABLATION_ENRICHED_WITH_COUNTS: INITIAL_FEATURE_BLOCKS + ['precision', 'thresholds', 'implied', 'term'],
    # Après réglage, retirer les fréquences géographiques améliore légèrement CV et score public : tester la deuxième série sur cette base
    ABLATION_CORE_WITHOUT_COUNTS: CORE_FEATURE_BLOCKS,
    ABLATION_ENRICHED: CORE_FEATURE_BLOCKS + ['precision', 'thresholds', 'implied', 'term'],
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Variantes inconnues dans RUN_ONLY : {sorted(unknown)}')
