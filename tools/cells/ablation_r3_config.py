EXPERIMENT_ID = 'ablation_round3'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None             # Exemple : [ABLATION_ENRICHED_REFERENCE, ABLATION_ADD_PRODUCT_FLAGS] ; None sélectionne toutes les variantes
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / EXPERIMENT_ID

FAST_PARAMS = dict(
    iterations=1600, depth=8, learning_rate=0.05, l2_leaf_reg=3,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    use_best_model=False, allow_writing_files=False, verbose=False,
)
CORE_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing']
ENRICHED_FEATURE_BLOCKS = CORE_FEATURE_BLOCKS + ['precision', 'thresholds', 'implied', 'term']   # Meilleur jeu de variables de la deuxième série (72 colonnes)
PRODUCT_RISK_BLOCKS = ['purpose', 'ltv_consistency', 'risk_layers', 'tract_income']
VARIANTS = {
    ABLATION_ENRICHED_REFERENCE: ENRICHED_FEATURE_BLOCKS,
    ABLATION_ADD_PRODUCT_FLAGS: ENRICHED_FEATURE_BLOCKS + ['purpose'],
    ABLATION_ADD_LTV_CONSISTENCY: ENRICHED_FEATURE_BLOCKS + ['ltv_consistency'],
    ABLATION_ADD_RISK_INTERACTIONS: ENRICHED_FEATURE_BLOCKS + ['risk_layers'],
    ABLATION_ADD_TRACT_INCOME: ENRICHED_FEATURE_BLOCKS + ['tract_income'],
    ABLATION_ALL_PRODUCT_RISK: ENRICHED_FEATURE_BLOCKS + PRODUCT_RISK_BLOCKS,
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Variantes inconnues dans RUN_ONLY : {sorted(unknown)}')
