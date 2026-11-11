# Résultats des expériences

Toutes les expériences utilisent les mêmes données préparées et les mêmes cinq plis stratifiés (graine 42). Le score est l'AUC moyenne des cinq plis. Deux variantes ne sont comparées que si elles ont tourné dans la même session, pli par pli.

## Vue d'ensemble

| Étape | Notebook | AUC, cinq plis | Score public |
|---|---|---:|---:|
| Départ : CatBoost sur CPU, profondeur 8, 56 colonnes | 03 | 0.91604 | 0.91857 |
| + variables de la série 1 (LTV recalculé…) et réglage | 02, 03 | 0.91926 | 0.92073 |
| + variables de la série 2 (précision et seuils du LTV), trois modèles | 02, 04 | 0.92051 | 0.92190 |
| + trois graines par modèle | 04 | 0.92073 | 0.92197 |
| **+ réentraînement sur les 105 000 lignes (finale)** | 04 | — | **0.92301** |

## Données (notebooks 01 et 05)

- Validation adversariale apprentissage / test : AUC 0.501. Prédire la cible avec l'identifiant seul : 0.496. Pas de lien entre identifiants voisins.
- Les champs dérivés sont cohérents avec leurs champs d'origine. 13 lignes seulement sont en double sur toutes les variables.
- Les valeurs inhabituelles ne sont pas des erreurs à nettoyer : le modèle les prédit correctement.

| Groupe | Lignes | Taux d'acceptation | Probabilité moyenne prédite |
|---|---:|---:|---:|
| Toutes | 105 000 | 85 % | — |
| LTV absent | 11 874 | 54 % | 54 % |
| Population du secteur = 0 | 2 117 | 17 % | 17 % |
| Revenu de la zone = 0 | 1 330 | 24 % | 24 % |
| Prêt / valeur du bien > 2 | 148 | 14 % | 14 % |
| Revenu ≤ 0 | 954 | 56 % | 57 % |

## Modèles de départ (notebook 03)

| Modèle | Calcul | AUC |
|---|---|---:|
| CatBoost profondeur 8, 1 600 itérations | CPU | 0.91604 |
| CatBoost profondeur 6, 1 600 itérations | CPU | 0.91507 |
| CatBoost profondeur 8, mêmes paramètres | GPU | 0.91498 |
| Régression logistique | CPU | 0.90020 |
| XGBoost, réglage fixe | CPU | 0.89284 |
| Ridge / forêt aléatoire avec transformations numériques | CPU | 0.8849 / 0.8898 |
| AutoGluon (pli 1 seulement) | GPU | 0.8996 |

L'écart CPU / GPU s'explique sans doute par l'échantillonnage par défaut (MVS sur CPU, bayésien sur GPU).

## Réglage de CatBoost (notebook 03)

Early stopping sur 15 % de la partie d'apprentissage, puis réentraînement pendant `b / 0.85` itérations. Référence : profondeur 8 sur GPU, taux 0.05, 1 600 itérations.

| Réglage | AUC | Gain | Plis |
|---|---:|---:|---:|
| **profondeur 8, taux 0.03, L2 = 10, Bernoulli 0.8** | **0.91652** | +0.00156 | 5/5 |
| **profondeur 6, le reste identique** | 0.91638 | +0.00142 | 5/5 |
| profondeur 8, L2 = 30 | 0.91602 | +0.00106 | 5/5 |
| référence | 0.91495 | — | — |

Sur le pli 1, l'échantillonnage Bernoulli apporte à lui seul environ +0.001 ; la profondeur 10 fait moins bien.

Modes de calcul essayés ensuite (boosting ordonné, NewtonL2, Depthwise, Lossguide, Poisson, taux 0.66, 254 bins, sans combinaisons de catégories) : le mieux est 254 bins avec +0.00023 sur 4 plis, sous le critère.

## Variables (notebook 02)

Modèle rapide : profondeur 8, taux 0.05, L2 = 3, Bernoulli 0.8, 1 600 itérations.

**Série 1**, référence = 56 colonnes (0.91566)

| Bloc | AUC | Gain | Plis |
|---|---:|---:|---:|
| **Tous les blocs (68 colonnes)** | **0.91866** | **+0.00300** | 5/5 |
| **LTV recalculé** | 0.91775 | +0.00209 | 5/5 |
| Profil de valeurs manquantes | 0.91588 | +0.00022 | 4/5 |
| Ratios de revenu | 0.91577 | +0.00010 | 4/5 |
| Bornes du DTI | 0.91576 | +0.00009 | 4/5 |
| Fréquences géographiques | 0.91553 | −0.00013 | 1/5 |

**Série 2**, référence = tous les blocs de la série 1 (0.91810)

| Bloc | AUC | Gain | Plis |
|---|---:|---:|---:|
| **Tous les blocs de la série 2, sans les fréquences (72 colonnes)** | **0.91935** | **+0.00125** | 5/5 |
| Tous les blocs, avec les fréquences | 0.91932 | +0.00122 | 5/5 |
| **Précision du LTV** | 0.91884 | +0.00075 | 4/5 |
| **Seuils du LTV** | 0.91884 | +0.00074 | 5/5 |
| Durée | 0.91834 | +0.00025 | 3/5 |
| Sans les fréquences seulement | 0.91822 | +0.00012 | 4/5 |
| Valeur reconstituée | 0.91820 | +0.00010 | 3/5 |

Quand le LTV est déclaré en entier, 90.7 % des demandes sont acceptées, contre 84.4 % avec deux décimales.

**Série 3** (indicateurs de produit, cohérence du LTV, cumul de risques, revenu relatif au secteur) : au mieux +0.00035 sur 3 plis, rien n'est gardé.

## Modèle final (notebook 04)

| Modèle | AUC, graine 42 | Gain sur la référence | Plis |
|---|---:|---:|---:|
| **Moyenne des trois** | **0.92051** | +0.00120 | 4/5 |
| profondeur 6, jeu réduit (68 colonnes) | 0.92015 | +0.00085 | 4/5 |
| profondeur 8, jeu enrichi (72 colonnes) | 0.91999 | +0.00068 | 4/5 |
| profondeur 6, jeu enrichi | 0.91989 | +0.00058 | 4/5 |
| référence : profondeur 6, variables de la série 1 | 0.91931 | — | — |

Avec trois graines par modèle : **0.92073** (score public 0.92197).

**Courbe d'apprentissage** (profondeur 6, jeu réduit, mêmes lignes de validation)

| Pli | 67 200 lignes | 84 000 lignes | Gain |
|---|---:|---:|---:|
| 1 | 0.91285 | 0.91430 | +0.00145 |
| 2 | 0.92640 | 0.92742 | +0.00102 |
| 3 | 0.91994 | 0.92075 | +0.00081 |
| 4 | 0.91499 | 0.91629 | +0.00130 |
| 5 | 0.92013 | 0.92157 | +0.00144 |
| **Moyenne** | | | **+0.00120** |

**Soumissions**

| Soumission | Score public |
|---|---:|
| **Neuf modèles réentraînés sur les 105 000 lignes** | **0.92301** |
| Moitié réentraînement, moitié modèles de plis | 0.92259 |
| Moyenne des modèles de plis | 0.92197 |

## Ce qui n'a pas marché (notebook 05)

| Piste | Résultat |
|---|---|
| LightGBM (31 ou 127 feuilles) | 0.9111 seul ; en mélange à 10 % : +0.00000 ; à 20 et 30 % : moins bon |
| Réseau de neurones TabM-mini (pli 1) | 0.90878, soit 0.0061 sous CatBoost ; corrélation 0.984 ; aucun gain en mélange |
| Pseudo-étiquettes sur le test | +0.00008, 3 plis sur 5 |
| Mélange entre toutes les expériences, choisi par validation imbriquée | garde l'ensemble final seul dans 4 plis sur 5 |
| Résidus par groupe (rapport proche de 1 = rien à gagner ; groupes aléatoires : 1.02) | secteur 0.96, comté 0.97, aire urbaine 0.90, État 1.09 ; indicateurs de produit 1.77 sur 17 groupes seulement |

## Méthode

- **Critère** : gain moyen d'au moins 0.0003 (0.0005 pour les dernières expériences) et au moins 4 plis améliorés sur 5.
- **Score public** : en tirant 13 500 lignes parmi les prédictions hors pli, l'écart-type de l'AUC est de 0.0037 et celui de la différence entre deux soumissions de 0.0006. Deux soumissions à moins de 0.001 d'écart ne sont pas départagées par le score public.
- **Sessions GPU** : le même code donne jusqu'à 0.0005 d'écart par pli entre deux sessions Colab.
- Les modèles réentraînés sur toutes les lignes n'ont pas de score de validation. Leur choix repose sur la courbe d'apprentissage ; l'ordre des scores publics (0.92301 > 0.92259 > 0.92197) va dans le même sens.
