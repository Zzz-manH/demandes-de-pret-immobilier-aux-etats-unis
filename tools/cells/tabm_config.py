EXPERIMENT_ID = 'tabm'
FOLDS = [1]                 # Commencer par le fold 1 ; passer à [1, 2, 3, 4, 5] si le résultat le justifie
OUTPUT_DIR = ARTIFACTS / '05_error_analysis' / EXPERIMENT_ID
CATBOOST_DIR = ARTIFACTS / '04_final_model' / 'cv_models'   # Résultats des cinq folds du notebook 04

CORE_FEATURE_BLOCKS = ['dti', 'ltv', 'income', 'missing']
ENRICHED_FEATURE_BLOCKS = CORE_FEATURE_BLOCKS + ['precision', 'thresholds', 'implied', 'term']   # Les mêmes 72 colonnes que CatBoost final
NN_CONFIG = dict(
    k=32,                   # 32 MLP partageant la plupart de leurs poids
    width=512, depth=3, dropout=0.1,
    cat_dim=8,              # Dimension des embeddings par colonne catégorielle
    ple_bins=16,            # Nombre de segments de l’encodage linéaire par morceaux
    rare_min_count=5,       # Catégories présentes moins de 5 fois dans le train externe : « inconnu »
    lr=1e-3, weight_decay=1e-4, batch_size=512,
    max_epochs=60, patience=8,
    inner_stop_fraction=0.15,
)
BLEND_WEIGHTS = [0.1, 0.2, 0.3]
