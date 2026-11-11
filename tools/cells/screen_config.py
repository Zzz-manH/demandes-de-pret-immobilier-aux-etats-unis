EXPERIMENT_ID = 'training_modes'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None                 # Exemple : [TRAINING_STANDARD, TRAINING_ORDERED] ; None sélectionne toutes les variantes
INNER_STOP_FRACTION = 0.15
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / EXPERIMENT_ID

CORE_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing']
REDUCED_FEATURE_BLOCKS = CORE_FEATURE_BLOCKS + ['precision', 'thresholds']        # Les 68 variables de CatBoost profondeur 6 réduit
DEMO_DUPLICATES = [                                 # Codes démographiques et indicateurs d’observation redondants avec derived_*
    'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
    'applicant_race-1', 'co-applicant_race-1',
    'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed',
    'applicant_race_observed', 'co-applicant_race_observed',
    'applicant_sex_observed', 'co-applicant_sex_observed',
]
SCREEN_BASE = dict(              # Proche de CatBoost profondeur 6 réduit (CatBoost profondeur 6), avec un taux d’apprentissage doublé pour le tri ; le jeu externe sert uniquement à évaluer
    depth=6, learning_rate=0.06, l2_leaf_reg=10,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', eval_metric='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    iterations=6000, od_type='Iter', od_wait=200, use_best_model=True,
    allow_writing_files=False, verbose=False,
)
VARIANTS = {
    TRAINING_STANDARD: {},
    TRAINING_ORDERED: {'boosting_type': 'Ordered'},            # Sur GPU, Plain est le défaut au-delà de 50 000 lignes ; Ordered reste à tester
    TRAINING_NEWTON_SCORE: {'score_function': 'NewtonL2'},         # Valeur par défaut : Cosine
    TRAINING_DEPTHWISE: {'grow_policy': 'Depthwise', 'depth': 7, 'min_data_in_leaf': 50},
    TRAINING_LOSSGUIDE: {'grow_policy': 'Lossguide', 'depth': 10, 'max_leaves': 48,
                     'min_data_in_leaf': 50, 'score_function': 'NewtonL2'},
    TRAINING_POISSON: {'bootstrap_type': 'Poisson', 'subsample': 0.8},   # Disponible uniquement sur GPU
    TRAINING_ROWS066: {'subsample': 0.66},                      # Sous-échantillonnage des lignes plus marqué
    TRAINING_BINS254: {'border_count': 254},                 # Sur GPU, 128 seuils numériques par défaut
    TRAINING_NO_CATEGORY_COMBINATIONS: {'max_ctr_complexity': 1},                  # Sans combinaison de catégories, pour réduire la variance
    TRAINING_DROP_REDUNDANT_DEMOGRAPHICS: {'drop': DEMO_DUPLICATES},             # Supprimer les 10 colonnes démographiques redondantes
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Variantes inconnues dans RUN_ONLY : {sorted(unknown)}')
