# Demandes de prêt immobilier aux États-Unis : prédire l'acceptation

Projet du Data Challenge d'apprentissage supervisé. Il s'agit d'estimer la probabilité qu'une demande de prêt immobilier soit acceptée (`accepted = 1`), à partir de données HMDA. La métrique est l'AUC. Le jeu contient 105 000 lignes d'apprentissage, 45 000 de test et 56 variables (13 numériques, 43 catégorielles).

**Résultat.** Soumission finale : [submissions/20261016_FINAL_full_Bseeds.csv](submissions/20261016_FINAL_full_Bseeds.csv), score public **0.92301**. Notre première soumission était à 0.91857.

Le détail des expériences est dans [RESULTS.md](RESULTS.md) et l'historique des soumissions dans [submissions/README.md](submissions/README.md).

## Les cinq notebooks

| Notebook | Contenu |
|---|---|
| [01_data_exploration](01_data_exploration.ipynb) | contrôle des CSV, exploration, préparation des 56 variables, plis fixes |
| [02_feature_engineering](02_feature_engineering.ipynb) | trois séries de variables et leurs ablations |
| [03_model_selection](03_model_selection.ipynb) | modèles de départ, transformations numériques, réglage de CatBoost |
| [04_final_model](04_final_model.ipynb) | modèles sur cinq plis, graines, courbe d'apprentissage, réentraînement complet, soumissions |
| [05_error_analysis](05_error_analysis.ipynb) | résidus, LightGBM, réseau de neurones, mélanges, pseudo-étiquettes, variabilité des scores |

Les notebooks sont enregistrés avec leurs sorties : on peut lire les tableaux et les figures sans rien exécuter.

## La solution en bref

Neuf modèles CatBoost sur GPU, entraînés sur les 105 000 lignes et moyennés à poids égaux : trois réglages × trois graines (42, 2027, 31415).

| Modèle | Paramètres | Variables |
|---|---|---|
| profondeur 6, jeu enrichi | profondeur 6, taux 0.03, L2 = 10, Bernoulli 0.8 | 56 d'origine + bornes du DTI, LTV recalculé, ratios de revenu, valeurs manquantes, précision et seuils du LTV, valeur reconstituée, durée (72 colonnes) |
| profondeur 8, jeu enrichi | profondeur 8, le reste identique | les mêmes |
| profondeur 6, jeu réduit | comme le premier | les mêmes sans la valeur reconstituée ni la durée (68 colonnes) |

Le gain vient de trois choses : les variables autour du LTV (environ +0.004), le réglage de CatBoost (+0.0015) et le réentraînement sur toutes les lignes (+0.001). Le reste de ce que nous avons essayé n'a pas apporté de gain régulier.

## Données

Les données ne sont pas dans le dépôt. Télécharger `train.csv`, `test.csv` et `sample_submission.csv` depuis la [page Kaggle de la compétition](https://www.kaggle.com/competitions/demandes-de-pret-immobilier-aux-etats-unis/data) et les placer dans `data/`.

Nous n'utilisons pas les données HMDA publiques : le test en vient.

## Exécution

- **Lire seulement** : ouvrir les notebooks, les sorties sont déjà là.
- **Réexécuter sans entraîner** : par défaut tous les boutons `TRAIN` sont à `False` et les notebooks relisent les résultats sauvegardés dans `artifacts/` (environ 290 Mo, hors dépôt). Le notebook 01 entraîne quand même un petit modèle de contrôle sur CPU et le notebook 02 réécrit les variables. Sans `artifacts/`, le notebook 01 se réexécute entièrement à partir des CSV et recrée les données préparées, et le notebook 02 reconstruit les variables (sections 1 à 5) ; ses tableaux d'ablation et les notebooks 03 à 05 s'arrêtent ensuite faute de résultats.
- **Tout refaire depuis les CSV** : exécuter 01, puis les sections 1 à 5 de 02 ; dans 04, activer `cv_models`, `learning_curve` et `full_refit` (GPU T4). Les notebooks 03 et 05 ne sont pas nécessaires pour la soumission finale.
- **Sur Colab** : placer le dossier du projet dans `MyDrive/Colab Notebooks/`, ou définir `HMDA_PROJECT_DIR`.

Dépendances : [requirements.txt](requirements.txt). CatBoost est fixé à la version 1.2.10.

## Organisation

```text
01–05 *.ipynb      les cinq notebooks
data/              CSV de la compétition (à télécharger)
artifacts/         données préparées, variables, prédictions et résultats (hors dépôt)
submissions/       fichiers soumis et historique des scores
tools/             code des cellules, textes et générateur des notebooks
RESULTS.md         résultats des expériences
```

Les premiers notebooks de travail, avant réorganisation, sont dans le premier commit du dépôt.
