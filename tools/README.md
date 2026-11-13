# tools

- `build_notebooks.py` assemble les six notebooks à partir du code de `cells/` et des textes de `textes/`.
- `cells/*.py` : le code des expériences, une cellule par fichier. `naming_compatibility.py` fait le lien entre les noms courts utilisés dans les fichiers de résultats et les libellés affichés.
- `textes/*.md` : le texte des notebooks, les cellules étant séparées par `<!-- cellule -->`.

Pour régénérer les notebooks :

```bash
python tools/build_notebooks.py /tmp/notebooks
```

Les notebooks générés n'ont pas de sorties. Ceux de la racine ont été exécutés ; générer dans un autre dossier pour ne pas les écraser.

| Préfixe des cellules | Contenu |
|---|---|
| `data_contract`, `helpers`, `dti_bounds`, `features_round1`–`3` | parties communes : contrôle des données préparées, fonctions utiles, variables |
| `prep_` | préparation et contrôle des données (notebook 01) |
| `ablation_` | ablations des trois séries de variables (notebook 02) |
| `tuning_`, `screen_` | réglage et modes de calcul de CatBoost (notebook 03) |
| `cv_`, `refit_` | modèles sur cinq plis, courbe d'apprentissage, réentraînement complet (notebook 04) |
| `lgbm_`, `tabm_`, `pseudo_` | LightGBM, réseau de neurones, pseudo-étiquettes (notebook 05) |
| `interp_` | valeurs de Shapley, calibration, groupes, modèle sans variables protégées (notebook 06) |
