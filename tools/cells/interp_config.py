EXPERIMENT_ID = 'protected_ablation'
FOLDS = [1, 2, 3, 4, 5]
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = ARTIFACTS / '06_interpretation' / EXPERIMENT_ID

GPU_BASE = dict(
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    allow_writing_files=False, verbose=False,
)
# Le modèle « profondeur 6, jeu réduit » du notebook 04 (68 colonnes)
MODEL_PARAMS = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
MODEL_BLOCKS = ['dti', 'ltv', 'income', 'missing', 'precision', 'thresholds']

# Origine, ethnicité, sexe et âge de l'emprunteur et du coemprunteur
PROTECTED_COLUMNS = [
    'derived_ethnicity', 'derived_race', 'derived_sex',
    'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
    'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed',
    'applicant_race-1', 'co-applicant_race-1',
    'applicant_race_observed', 'co-applicant_race_observed',
    'applicant_sex', 'co-applicant_sex',
    'applicant_sex_observed', 'co-applicant_sex_observed',
    'applicant_age', 'co-applicant_age',
]
REFERENCE = 'reference'
WITHOUT_PROTECTED = 'sans_protegees'
VARIANTS = {REFERENCE: [], WITHOUT_PROTECTED: PROTECTED_COLUMNS}    # colonnes retirées
VARIANT_LABELS = {REFERENCE: 'Référence (68 colonnes)',
                  WITHOUT_PROTECTED: f'Sans variables protégées ({68 - len(PROTECTED_COLUMNS)} colonnes)'}

SHAP_FOLD = 1          # les valeurs de Shapley viennent du modèle de référence de ce pli
SHAP_ROWS = 5000       # lignes de validation tirées au hasard
SHAP_PATH = OUTPUT_DIR / f'{REFERENCE}__fold{SHAP_FOLD}__shap.parquet'

if BLOCKS is not None:
    full_train, _, full_cats = variant_frames(MODEL_BLOCKS)
    require(set(PROTECTED_COLUMNS) <= set(full_train.columns), 'Variable protégée absente des colonnes')
