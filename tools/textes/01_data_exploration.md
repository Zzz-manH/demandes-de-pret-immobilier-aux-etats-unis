# 01 · Lecture et exploration des données

On veut estimer la probabilité qu'une demande de prêt immobilier soit acceptée (`accepted = 1`), à partir de données HMDA américaines. La métrique est l'AUC. Il y a 105 000 lignes d'apprentissage, 45 000 de test et 56 variables.

Ce notebook vérifie les CSV, explore les données et écrit dans `artifacts/01_data_exploration/` une version préparée commune à tous les notebooks : les 56 variables typées, la cible, les identifiants et cinq plis fixes. Aucune ligne n'est supprimée et la cible ne sert à construire aucune variable.

{NOTEBOOK_MAP}

Si les fichiers préparés existent déjà, ils sont vérifiés et non réécrits. La section 8 entraîne un petit modèle sur CPU.
<!-- cellule -->
## 1. Lecture en texte, identifiants et cible

Tout est d'abord lu comme du texte, pour ne pas perdre les zéros en tête des codes (`00123`). On vérifie ensuite l'ordre des colonnes, les identifiants, le fichier de soumission modèle, et que la cible vaut bien `0` ou `1`.
<!-- cellule -->
## 2. Types et règles de préparation

13 variables sont numériques : montant du prêt et valeur du bien en dollars, revenu en milliers de dollars par an, LTV en pourcentage. Les 43 autres sont traitées comme des catégories, y compris les codes numériques, les intervalles de DTI et d'âge et les codes géographiques.

| Cas | Traitement |
|---|---|
| nombre vide | `NaN` ; erreur si le texte n'est pas un nombre |
| catégorie vide | modalité `MISSING` |
| code entier écrit `1.0` | ramené à `1` |
| `county_code`, `derived_msa-md` | chaîne de cinq chiffres ; le code spécial `99999` est gardé |
| « non applicable », « pas de co-emprunteur », exemption (`1111`)… | codes gardés tels quels, distincts du vide |
| revenu négatif ou nul, LTV extrême, âge du bâti à −1 | valeurs gardées, comptées en section 7 |
<!-- cellule -->
## 3. Même préparation pour l'apprentissage et le test

La fonction ne voit pas la cible et n'ajuste rien sur les données (ni médiane, ni quantile, ni encodeur).
<!-- cellule -->
## 4. Contrôle : mêmes valeurs, même ordre des lignes
<!-- cellule -->
## 5. Vue d'ensemble
<!-- cellule -->
## 6. Variables liées à l'acceptation

Le taux d'acceptation global est d'environ 85 % (trait pointillé). Il est nettement plus bas quand il n'y a pas eu de système de décision automatisé (`aus-1 = 6`), quand le DTI est élevé, et quand le LTV est élevé ou absent.
<!-- cellule -->
**Valeurs manquantes.** Elles ne sont pas réparties au hasard : le taux d'acceptation change selon qu'un champ est renseigné ou non. On les garde donc telles quelles ; CatBoost les gère directement (`nan_mode=Min`).
<!-- cellule -->
## 7. Valeurs inhabituelles

Revenus négatifs ou nuls, très gros montants, LTV au-dessus de 100 %, lignes en double sur toutes les variables : on les compte, on ne les supprime pas.
<!-- cellule -->
## 8. Contrôles de qualité

- **Validation adversariale** : un modèle essaie de distinguer apprentissage et test. Une AUC proche de 0.5 veut dire qu'il n'y arrive pas.
- **Identifiants** : sont-ils liés à la cible, deux identifiants voisins se ressemblent-ils ?
- **Champs dérivés** : sont-ils cohérents avec les champs dont ils viennent ?
- **Valeurs inhabituelles** : quel taux d'acceptation dans ces groupes ?
<!-- cellule -->
Apprentissage et test se ressemblent, les identifiants ne portent pas de signal et les champs dérivés sont cohérents. Les groupes aux valeurs inhabituelles sont moins souvent acceptés, ce qui est une information utile : on garde ces lignes.
<!-- cellule -->
## 9. Cinq plis fixes

`StratifiedKFold(5, shuffle=True, random_state=42)`. Toutes les expériences utilisent ce découpage, ce qui permet de comparer deux variantes pli par pli.
<!-- cellule -->
## 10. Enregistrement

`train_features.csv` et `test_features.csv` contiennent les 56 variables ; d'autres fichiers gardent la cible, les identifiants et les plis. `schema.json` décrit les types et `manifest.json` les empreintes SHA-256, que les notebooks suivants vérifient avant de commencer. Les fichiers existants ne sont pas réécrits, sauf si `REBUILD_PREPARED` est activé.
<!-- cellule -->
## 11. Relecture

On relit les fichiers enregistrés et on les compare aux données recalculées : valeurs, catégories, ordre des colonnes, identifiants, plis, empreintes.
