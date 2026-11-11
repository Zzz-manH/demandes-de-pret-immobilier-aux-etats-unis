# 05 · Analyse des erreurs et pistes abandonnées

Ce qu'on a cherché une fois le modèle en place, et ce qui n'a rien donné :

1. **Résidus** : y a-t-il des groupes que le modèle surestime ou sous-estime ?
2. **Autres modèles** : LightGBM et un réseau de neurones complètent-ils CatBoost ?
3. **Mélanges** entre toutes nos expériences.
4. **Pseudo-étiquettes** : les 45 000 lignes de test peuvent-elles servir à l'entraînement ?
5. **Variabilité** : que vaut un petit écart de score ?

{NOTEBOOK_MAP}
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 0. Chargement des données et des variables
<!-- cellule -->
## 1. Résidus par groupe

Avec les prédictions hors pli du modèle « profondeur 6, jeu réduit », on calcule pour chaque groupe **[Σ(y − p)]² / Σ p(1 − p)**, puis la moyenne sur les groupes. Si le modèle est bien calibré dans chaque groupe, cette valeur est proche de 1 ; on la compare à celle de groupes tirés au hasard.
<!-- cellule -->
La géographie, le profil de l'établissement, type de prêt × DTI et LTV × DTI donnent des valeurs proches de celles des groupes aléatoires. Les indicateurs de produit sont plus dispersés (1.77, mais sur 17 groupes seulement). C'est ce qui a motivé la troisième série de variables du notebook 02, sans succès.
<!-- cellule -->
### Valeurs inhabituelles

Dans les groupes aux valeurs inhabituelles repérés dans le notebook 01, la probabilité moyenne prédite est proche du taux d'acceptation observé. Le modèle les traite donc correctement et il n'y a pas de raison de les retirer.
<!-- cellule -->
## 2. Autres modèles

### 2.1 LightGBM

Mêmes 72 variables, 31 ou 127 feuilles, même protocole que CatBoost. On regarde l'AUC sur cinq plis, puis des mélanges avec notre ensemble CatBoost à poids fixés (0 = CatBoost seul, 1 = LightGBM seul).
<!-- cellule -->
LightGBM fait environ 0.911, soit 0.009 de moins que CatBoost, et ses prédictions sont très corrélées aux siennes. Avec un poids de 10 % le score ne bouge pas, au-delà il baisse. Abandonné.
<!-- cellule -->
### 2.2 Réseau de neurones (TabM-mini)

Une version simplifiée de TabM (Gorishniy et al.) écrite en PyTorch : 32 réseaux à trois couches qui partagent leurs poids, chacun avec son propre vecteur de signes en entrée. Variables numériques : transformation par quantiles, encodage linéaire par 16 segments, indicateurs de valeurs manquantes. Catégories : embeddings de dimension 8. On s'arrête après le pli 1 si l'AUC est sous 0.915 environ ou si le mélange n'apporte rien.
<!-- cellule -->
0.9088 sur le pli 1, soit 0.006 de moins que CatBoost, avec une corrélation de 0.984. Aucun gain en mélange avec des poids de 10 à 30 %. Les quatre autres plis n'ont pas été lancés.
<!-- cellule -->
## 3. Mélanges entre expériences

Candidats : notre ensemble final, la référence, LightGBM et les premiers modèles sur CPU (CatBoost profondeur 6 et 8, XGBoost). Sur quatre plis, on construit un mélange pas à pas (moyenne des logits, répétitions permises) ; on l'évalue sur le cinquième. On recommence pour chaque pli, afin que le choix du mélange et son évaluation ne se fassent pas sur les mêmes lignes.
<!-- cellule -->
Les anciens modèles sont moins bons et très corrélés à l'ensemble final. La sélection garde presque toujours l'ensemble final seul.
<!-- cellule -->
## 4. Pseudo-étiquettes

Dans chaque pli, un « professeur » (la moyenne de nos neuf modèles, entraînés sur la partie d'apprentissage du pli) prédit les 45 000 lignes de test. Un « élève » (profondeur 6, jeu réduit) est ensuite entraîné sur la partie d'apprentissage plus le test avec ces probabilités comme cibles (`CrossEntropy`). L'early stopping n'utilise que des lignes à vraie cible. On compare à un modèle sans pseudo-étiquettes, dans la même session et sur le même pli.
<!-- cellule -->
+0.00008, sur 3 plis seulement : sous le critère. À titre de comparaison, 25 % de vraies cibles en plus font gagner 0.0012 (notebook 04), alors que 54 % de lignes en plus avec des cibles prédites ne font rien gagner. Ce qui manque, ce sont de vraies étiquettes.
<!-- cellule -->
## 5. Que vaut un petit écart ?

On imite le classement public en calculant l'AUC sur 13 500 lignes tirées au hasard parmi les prédictions hors pli. On compare aussi un même réglage lancé dans deux sessions Colab.
<!-- cellule -->
D'un tirage à l'autre, l'AUC d'une soumission varie d'environ 0.004 (écart-type), mais la différence entre l'ensemble final et la référence varie seulement de 0.0006. Avec un écart de 0.0014, l'ordre s'inverse dans 1 % des tirages environ ; avec 0.0005, dans 20 %. Entre deux sessions Colab, l'écart par pli va de 0 à 0.0004. C'est pour cela que le critère des dernières expériences est de 0.0005.
<!-- cellule -->
## 6. Bilan

- Aucun contrôle ne justifiait de supprimer des lignes. Le gain des variables vient surtout du LTV.
- Ni les résidus par groupe, ni les autres modèles, ni les modes de calcul, ni les mélanges, ni les pseudo-étiquettes n'ont apporté de gain assez régulier.
- Ce qui aide, c'est davantage de données étiquetées : d'où le réentraînement sur les 105 000 lignes.

Nous n'utilisons pas les données HMDA publiques : le test en vient, et les rapprocher reviendrait à récupérer ses cibles.
