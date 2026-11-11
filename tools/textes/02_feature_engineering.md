# 02 · Construction des variables

L'idée est d'ajouter des variables qu'un arbre retrouve mal tout seul avec des seuils, sans toucher aux 56 colonnes d'origine. Aucune n'utilise la cible.

Trois séries :

1. bornes numériques du DTI, LTV recalculé, ratios de revenu, profil de valeurs manquantes, fréquences géographiques ;
2. précision de saisie du LTV, seuils de LTV selon le type de prêt, valeur du bien reconstituée, durée ;
3. idées venues de l'analyse des résidus (notebook 05) : indicateurs de produit, cohérence du LTV, cumul de risques, revenu relatif au secteur.

Chaque série est testée par **ablation**, avec un CatBoost rapide et les cinq plis fixes. On garde un bloc si le gain moyen d'AUC atteint le seuil et s'il est positif sur au moins 4 plis sur 5. Toutes les variables sont enregistrées dans `artifacts/02_feature_engineering/feature_store/` pour les notebooks suivants.

{NOTEBOOK_MAP}
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 1. Chargement des données préparées

Vérification des fichiers du notebook 01 : empreintes, ordre des colonnes, identifiants, plis.
<!-- cellule -->
## 2. Première série

### Bornes du DTI

Le DTI est donné par intervalles (`30%-<36%`). On en tire deux colonnes numériques, la borne basse et la borne haute, en gardant la catégorie d'origine. Une borne inconnue reste `NaN`.
<!-- cellule -->
### Les blocs

Un ratio n'est calculé que si son dénominateur est strictement positif ; sinon il est manquant.

| Bloc | Colonnes | Idée |
|---|---|---|
| `dti` | `dti_lower`, `dti_upper` | bornes numériques du DTI |
| `ltv` | `ltv_calc`, `ltv_gap` | LTV recalculé (prêt / valeur du bien) et écart avec le LTV déclaré |
| `income` | `income_to_area`, `loan_to_income`, `value_to_income` | revenu par rapport à la zone ; prêt et valeur du bien par rapport au revenu |
| `missing` | `missing_pattern`, `missing_count` | quels champs manquent, et combien |
| `freq` | `freq_county_code`, `freq_derived_msa-md`, `freq_state_code` | fréquence du comté, de l'aire urbaine et de l'État dans apprentissage + test |
<!-- cellule -->
## 3. Deuxième série

`ltv_precision` regarde le **texte** du LTV dans le CSV (`80.0` ou `79.85`), d'où la relecture de cette colonne en chaîne.

| Bloc | Colonnes | Pourquoi |
|---|---|---|
| `precision` | `ltv_precision` (absent, entier, 1 à 3 décimales, plus) | 90.7 % d'acceptation pour un LTV entier contre 84.4 % avec deux décimales : la façon de déclarer dépend sans doute de l'établissement |
| `thresholds` | `ltv_band`, `ltv_over_program` | position par rapport à 80, 95, 97 et 100 ; écart au plafond habituel du type de prêt (classique 97, FHA 96.5, VA et RHS 100) |
| `implied` | `implied_property_value`, `implied_value_ratio` | valeur du bien déduite du prêt et du LTV, comparée à la valeur déclarée |
| `term` | `term_is_360`, `term_years` | durée en années, indicateur de prêt sur 30 ans |
<!-- cellule -->
## 4. Troisième série

| Bloc | Colonnes | Pourquoi |
|---|---|---|
| `purpose` | `product_flags` | sept indicateurs de produit regroupés ; c'est là que les résidus étaient les plus dispersés |
| `ltv_consistency` | `ltv_consistency`, `ltv_outside_interval` | le LTV déclaré est-il compatible avec des montants arrondis à 10 000 dollars ? |
| `risk_layers` | `risk_layer_count`, `risk_layer_pattern` | cumul de facteurs : LTV ou DTI élevé, pas de système automatisé, résidence non principale, LTV absent, second rang |
| `tract_income` | `income_to_tract` | revenu par rapport à la médiane du secteur de recensement |
<!-- cellule -->
## 5. Enregistrement des variables

Les notebooks 03 à 05 relisent ces fichiers et vérifient leurs empreintes. Cette cellule écrit les fichiers même si tous les boutons `TRAIN` sont à `False`.
<!-- cellule -->
## 6. Ablations

**Modèle rapide** : CatBoost sur GPU, profondeur 8, taux d'apprentissage 0.05, L2 = 3, Bernoulli 0.8, 1 600 itérations. Chaque variante ajoute un ou plusieurs blocs à la référence. Dans les tableaux, `delta_vs_base` est le gain moyen, `folds_better` le nombre de plis améliorés, `keep` indique si le critère est rempli.
<!-- cellule -->
### 6.1 Première série (référence : les 56 colonnes d'origine)

Critère : gain moyen d'au moins 0.0003, positif sur au moins 4 plis sur 5.
<!-- cellule -->
Tous les blocs ensemble gagnent **0.0030, sur les 5 plis**. Presque tout vient du LTV recalculé : **+0.0021** à lui seul. Les autres blocs changent peu le score pris un par un, et les fréquences géographiques le font légèrement baisser.
<!-- cellule -->
### 6.2 Deuxième série (référence : tous les blocs de la première)

Même critère.
<!-- cellule -->
Avec tous les blocs de la deuxième série et sans les fréquences géographiques (72 colonnes), le gain est de **0.00125, sur les 5 plis**. Pris seuls : précision du LTV +0.00075 (4 plis sur 5), seuils +0.00074 (5 sur 5). La valeur reconstituée et la durée ne passent pas le critère seules.
<!-- cellule -->
### 6.3 Troisième série (référence : les 72 colonnes)

Entre deux sessions Colab, le même modèle peut varier de quelques dix-millièmes. Le critère passe donc à 0.0005.
<!-- cellule -->
Le meilleur candidat gagne 0.00035 sur 3 plis seulement. Aucun bloc ne passe le critère : on n'ajoute rien.

## 7. Jeux de variables gardés

| Jeu | Contenu | Colonnes | Utilisé par |
|---|---|---:|---|
| **enrichi** | 56 d'origine + `dti`, `ltv`, `income`, `missing`, `precision`, `thresholds`, `implied`, `term` | 72 | CatBoost profondeur 6 et profondeur 8 |
| **réduit** | le même sans `implied` ni `term` | 68 | CatBoost profondeur 6 |
| série 1 | 56 d'origine + tous les blocs de la première série | 68 | référence |

Le jeu réduit enlève seulement les variables *ajoutées* sur la durée ; la colonne d'origine `loan_term` est toujours là.
