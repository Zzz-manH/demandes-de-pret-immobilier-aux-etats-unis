reduced_train, reduced_test, reduced_cats = variant_frames(REDUCED_FEATURE_BLOCKS)


def frames_for(spec):
    drop = spec.get('drop', [])
    unknown = set(drop) - set(reduced_train.columns)
    require(not unknown, f'Colonnes à supprimer introuvables : {sorted(unknown)}')
    columns = [c for c in reduced_train.columns if c not in drop]
    return reduced_train[columns], [c for c in reduced_cats if c not in drop]


def params_for(spec, **extra):
    params = dict(SCREEN_BASE)
    params.update({k: v for k, v in spec.items() if k != 'drop'})
    params.update(extra)
    return params


# Précontrôle : 20 itérations sur 3 000 lignes par variante pour détecter les paramètres incompatibles avec le GPU
rng = np.random.default_rng(SEED)
probe_rows = rng.choice(len(y), 3000, replace=False)
failed = {}
for name, spec in VARIANTS.items():
    train, cats = frames_for(spec)
    try:
        model = catboost.CatBoostClassifier(**params_for(spec, iterations=20))
        model.fit(train.iloc[probe_rows[:2500]], y.iloc[probe_rows[:2500]], cat_features=cats,
                  eval_set=(train.iloc[probe_rows[2500:]], y.iloc[probe_rows[2500:]]))
        checked_probabilities(model, train.iloc[probe_rows[2500:]])
        print(f'Précontrôle réussi : {name}({train.shape[1]} colonnes)')
    except Exception as exc:
        failed[name] = repr(exc)
        print(f'Précontrôle échoué, variante ignorée : {name} → {exc!r}')
require(TRAINING_STANDARD not in failed, 'Échec du précontrôle de Boosting standard ; vérifier l’environnement')
