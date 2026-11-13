# 06 · Interprétation du modèle

Les notebooks 01 à 05 cherchent le meilleur score. Celui-ci regarde ce que le modèle a appris :

1. **Quelles variables** pèsent dans la prédiction, et dans quel sens ?
2. **Calibration** : une probabilité de 0.9 correspond-elle à 90 % d'acceptations ?
3. **Par groupe** (origine, ethnicité, sexe, âge) : le modèle est-il aussi juste pour chacun ?
4. **Sans les variables protégées** : que devient le score, et que deviennent les écarts entre groupes ?

Un seul modèle est étudié, « profondeur 6, jeu réduit » (68 colonnes), l'un des trois réglages de l'ensemble final. Les résultats sont dans `artifacts/06_interpretation/`.
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 0. Chargement des données et des variables
<!-- cellule -->
## 1. Deux modèles entraînés dans la même session

- **Référence** : profondeur 6, jeu réduit, 68 colonnes, comme dans le notebook 04.
- **Sans variables protégées** : le même sans les 17 colonnes qui décrivent l'origine, l'ethnicité, le sexe et l'âge de l'emprunteur et du co-emprunteur (51 colonnes).

Même protocole que le notebook 04 : cinq plis, arrêt anticipé sur 15 % des lignes d'apprentissage, puis réentraînement. Les deux variantes d'un pli se suivent dans la même session. Les valeurs de Shapley sont calculées par CatBoost sur 5 000 lignes de validation du pli 1, avec le modèle de référence de ce pli.
<!-- cellule -->
## 2. Quelles variables comptent ?

La valeur de Shapley d'une variable pour une ligne est sa contribution à la prédiction, en log-odds. On prend la moyenne des valeurs absolues, par variable puis par famille de variables.
<!-- cellule -->
- L'**instruction du dossier** pèse le plus (24 % pour 6 colonnes) : système automatique de souscription utilisé (`aus-1`), pré-approbation, dépôt du dossier. Viennent ensuite l'endettement et le revenu (17 %), le produit (16 %), le LTV et la valeur du bien (14 %), la géographie (13 %).
- Le DTI est la première variable (8.5 %). Aucune ne domine : les cinq premières font 32 % du total, les dix premières 47 %.
- Les **17 variables protégées** font 12 % du total. La première, l'origine de l'emprunteur (`applicant_race-1`), est 13e ; son âge est 17e.
- `missing_pattern` est 5e : la liste des champs non renseignés d'un dossier est très informative.
- Le LTV agit par seuils : effet positif et stable jusqu'à 75–80 %, négatif au-delà, et chute nette autour de 100 %. C'est ce que les variables de précision et de seuils du notebook 02 cherchaient à capter. Un bien de moins de 200 000 $ environ pénalise le dossier ; au-delà de 400 000 $ la valeur ne joue plus.

Ces valeurs décrivent le modèle d'un seul pli, pas l'ensemble final, et une contribution n'est pas une cause : une variable peut porter l'information d'une autre qui manque dans les données (l'historique de crédit, par exemple).
<!-- cellule -->
## 3. Calibration

L'AUC ne mesure que l'ordre des dossiers. On vérifie ici que les probabilités elles-mêmes sont justes : les prédictions hors pli sont découpées en 20 tranches de même taille, et on compare dans chacune la probabilité moyenne au taux d'acceptation observé.
<!-- cellule -->
Le modèle est bien calibré sans aucune correction : l'erreur de calibration est de 0.0020 et l'écart le plus grand dans une tranche de 0.0086. Le score de Brier est de 0.052, contre 0.128 en prédisant le taux moyen pour tout le monde. L'histogramme montre aussi où sont les erreurs : la plupart des refus reçoivent une probabilité très basse, mais il en reste à tous les niveaux, y compris au-dessus de 0.9.
<!-- cellule -->
## 4. Par groupe

Pour chaque groupe d'au moins 500 lignes : taux d'acceptation observé, probabilité moyenne prédite (hors pli), écart entre les deux, et AUC calculée dans le groupe.
<!-- cellule -->
- **Les taux d'acceptation diffèrent beaucoup d'un groupe à l'autre** : 86.8 % pour les emprunteurs blancs, 73.9 % pour les emprunteurs noirs ; 86.5 % pour les non-hispaniques, 79.3 % pour les hispaniques ; 84.5 % pour les hommes seuls, 80.5 % pour les femmes seules. Ces écarts sont dans les données.
- **Le modèle les reproduit sans en ajouter** : dans chaque groupe la probabilité moyenne est à moins de 0.4 point du taux observé. Seule exception, les emprunteurs amérindiens (810 lignes), sous-estimés de 1.2 point.
- **La qualité du classement varie davantage** : l'AUC va de 0.883 (emprunteurs asiatiques) à 0.937, et de 0.904 pour les hommes seuls à 0.933 pour les femmes seules. Une partie de ces différences vient du taux d'acceptation de chaque groupe, qui change la difficulté du problème.
<!-- cellule -->
## 5. Sans les variables protégées

On compare les deux modèles pli par pli, puis les probabilités moyennes qu'ils donnent à chaque groupe.
<!-- cellule -->
- **Le score baisse de 0.0020**, sur les cinq plis (de −0.0011 à −0.0031) : quatre fois notre critère de 0.0005. Les variables protégées apportent donc une information que les 51 autres colonnes ne contiennent pas entièrement.
- **Les écarts entre groupes restent presque les mêmes.** Entre emprunteurs blancs et noirs, l'écart de probabilité moyenne passe de 12.9 à 10.1 points ; entre non-hispaniques et hispaniques, de 7.2 à 6.6 points ; entre hommes et femmes seuls, il ne diminue pas (4.2 puis 4.4 points). Le modèle retrouve l'essentiel des écarts à partir du reste du dossier : DTI, LTV, type de prêt, secteur géographique.
- **Le modèle sans ces variables n'est plus calibré par groupe** : il surestime de 2.4 points l'acceptation des emprunteurs noirs (76.2 % prédits, 73.9 % observés) et sous-estime de 0.5 point celle des emprunteurs blancs.
- Les deux modèles classent les dossiers de façon proche (corrélation de Spearman 0.957), avec un écart moyen de 0.014 sur la probabilité.

Limites : `missing_pattern` indique encore si l'âge est renseigné, et les variables de secteur (`tract_minority_population_percent` en particulier) sont liées à l'origine. Le modèle « sans variables protégées » n'est donc pas aveugle à ces caractéristiques.
<!-- cellule -->
## 6. Bilan

- Le modèle s'appuie d'abord sur l'instruction du dossier, puis sur le DTI et le LTV, avec des effets de seuil cohérents avec les variables construites dans le notebook 02.
- Ses probabilités sont calibrées, globalement et dans chaque groupe.
- Les variables protégées représentent 12 % des contributions et 0.0020 d'AUC. Pour la compétition, nous les gardons : seul le score compte.
- Dans un usage réel, les retirer ne suffirait pas : le score baisse, les écarts entre groupes restent, et la calibration par groupe se dégrade. Ces écarts viennent des décisions passées que le modèle apprend à reproduire, pas d'une variable en particulier.
- La référence a tourné ici dans une troisième session (GPU T4) : 0.92009, contre 0.92015 dans le notebook 04, soit moins que la variabilité entre deux sessions.
