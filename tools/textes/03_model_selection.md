# 03 · Choix et réglage des modèles

Quelle famille de modèles, puis quels paramètres pour CatBoost. Toutes les comparaisons utilisent les plis fixes du notebook 01.

1. Modèles de départ : régression logistique, XGBoost, CatBoost sur CPU et GPU, AutoGluon.
2. Ridge et forêt aléatoire avec des transformations numériques.
3. Réglage de CatBoost : taux d'apprentissage, profondeur, L2, échantillonnage.
4. Modes de calcul de CatBoost.

LightGBM et un réseau de neurones sont essayés en complément de CatBoost dans le notebook 05.

{NOTEBOOK_MAP}
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 0. Chargement des données et des variables
<!-- cellule -->
## 1. Modèles de départ

| Modèle | Réglage |
|---|---|
| Régression logistique | géographie simplifiée, interaction DTI × type de prêt, indicateurs de LTV extrême ; imputation et standardisation ajustées dans chaque pli |
| XGBoost | catégories natives, profondeur 6, early stopping dans le pli |
| CatBoost profondeur 6 et 8 | 56 colonnes d'origine, taux 0.05, L2 = 3, 1 600 itérations |

CatBoost est nettement devant. Avec les mêmes paramètres affichés, la version GPU fait environ 0.001 de moins que la version CPU ; l'explication la plus probable est l'échantillonnage par défaut, qui n'est pas le même sur CPU et sur GPU (voir la section 3). Le code de la régression logistique et de XGBoost date de nos premiers notebooks (premier commit du dépôt) ; le bloc ci-dessous permet de réentraîner les CatBoost.
<!-- cellule -->
**AutoGluon** (`extreme_quality`, GPU T4, 40 minutes, pli 1) : 0.8996. TabICL et TabDPT ont manqué de mémoire et seul un LightGBM a été entraîné, donc cet essai dit surtout que nos ressources ne suffisaient pas.
<!-- cellule -->
## 2. Ridge, forêt aléatoire et transformations

Six représentations des variables numériques sont essayées avec Ridge et une forêt aléatoire : valeurs brutes, polynômes de degré 2 et 3, quantiles, approximation RBF, splines ; les catégories sont en one-hot. Ridge gagne au mieux 0.003, la forêt rien. Les deux restent très loin de CatBoost (environ 0.916). Ce code date aussi de nos premiers notebooks.
<!-- cellule -->
## 3. Réglage de CatBoost

**Protocole.** Dans chaque pli, 15 % de la partie d'apprentissage sont réservés à l'early stopping, ce qui donne un nombre d'itérations *b*. Le modèle est ensuite réentraîné sur toute la partie d'apprentissage pendant `round(b / 0.85)` itérations, et le pli de validation ne sert qu'au score. La référence est le CatBoost profondeur 8 sur GPU (taux 0.05, 1 600 itérations). Douze réglages sont d'abord triés sur le pli 1, puis les meilleurs sont évalués sur les cinq plis.
<!-- cellule -->
Deux réglages se détachent : **profondeur 8** (taux 0.03, L2 = 10, Bernoulli 0.8), **+0.00156**, et le même en **profondeur 6**, **+0.00142**, tous deux sur les 5 plis. Sur le pli 1, l'échantillonnage Bernoulli apporte à lui seul environ 0.001, ce qui va dans le sens de cette explication de l'écart CPU/GPU. La profondeur 10 fait moins bien. On garde les deux profondeurs.
<!-- cellule -->
## 4. Modes de calcul de CatBoost

Sur le jeu réduit (68 colonnes), on essaie : boosting ordonné, score NewtonL2, croissance Depthwise ou Lossguide, échantillonnage de Poisson, taux d'échantillonnage de 0.66, 254 bins, pas de combinaisons de catégories, et le retrait de colonnes démographiques redondantes. Critère : 0.0005 de gain moyen et 4 plis sur 5.
<!-- cellule -->
Rien ne passe le critère. Le mieux est 254 bins, avec +0.00023 sur 4 plis. On reste sur le réglage de base : arbres symétriques, boosting Plain, score Cosine, Bernoulli 0.8.
<!-- cellule -->
## 5. Bilan

On garde **CatBoost sur GPU, en profondeur 8 et en profondeur 6**. Le notebook 04 les associe aux variables du notebook 02.
