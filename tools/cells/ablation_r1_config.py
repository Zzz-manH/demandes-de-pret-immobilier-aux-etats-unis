EXPERIMENT_ID = 'ablation_round1'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None             # Exemple : [ABLATION_ORIGINAL, ABLATION_ADD_LTV] ; None sélectionne toutes les variantes
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / EXPERIMENT_ID

FAST_PARAMS = dict(
    iterations=1600, depth=8, learning_rate=0.05, l2_leaf_reg=3,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    use_best_model=False, allow_writing_files=False, verbose=False,
)
VARIANTS = {
    ABLATION_ORIGINAL: [],
    ABLATION_ADD_DTI: ['dti'],
    ABLATION_ADD_LTV: ['ltv'],
    ABLATION_ADD_INCOME: ['income'],
    ABLATION_ADD_MISSINGNESS: ['missing'],
    ABLATION_ADD_GEOGRAPHIC_COUNTS: ['freq'],
    ABLATION_INITIAL: ['dti', 'ltv', 'income', 'missing', 'freq'],
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Variantes inconnues dans RUN_ONLY : {sorted(unknown)}')
