# 04 · Modèle final et soumission

Le modèle final est la moyenne de **trois réglages CatBoost × trois graines**, entraînés sur les **105 000 lignes**. Score public : **0.92301**.

| Modèle | Paramètres (notebook 03) | Variables (notebook 02) |
|---|---|---|
| profondeur 6, jeu enrichi | profondeur 6, taux 0.03, L2 = 10, Bernoulli 0.8 | 72 colonnes |
| profondeur 8, jeu enrichi | profondeur 8, le reste identique | 72 colonnes |
| profondeur 6, jeu réduit | comme le premier | 68 colonnes |
| référence | comme le premier | variables de la série 1 (68 colonnes) |

Trois étapes : validation des modèles et de leur moyenne sur cinq plis, courbe d'apprentissage, réentraînement sur toutes les lignes. Pour tout refaire sans les résultats sauvegardés : exécuter 01, puis les sections 1 à 5 de 02, puis activer ici `cv_models`, `learning_curve` et `full_refit`.

{NOTEBOOK_MAP}
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 0. Chargement des données et des variables
<!-- cellule -->
## 1. Modèles sur cinq plis et moyenne

Même protocole que dans le notebook 03 : early stopping sur 15 % de la partie d'apprentissage, puis réentraînement pendant `round(b / 0.85)` itérations. Chaque modèle prédit aussi le test. Les trois modèles finaux sont entraînés avec les graines 42, 2027 et 31415.
<!-- cellule -->
Les variables de la deuxième série apportent 0.0006 à 0.0009 selon le modèle. La moyenne des trois modèles donne 0.92051, et **0.92073** avec trois graines chacun, soit 0.0014 de plus que la référence, sur les 5 plis.
<!-- cellule -->
## 2. Courbe d'apprentissage

Un modèle de pli s'entraîne sur 84 000 lignes, soit 80 % des données. Est-ce que réentraîner sur les 105 000 lignes vaut la peine ? Pour en avoir une idée, on entraîne le modèle « profondeur 6, jeu réduit » sur 67 200 puis sur 84 000 lignes, avec les mêmes lignes de validation. Passer de 67 200 à 84 000, c'est 25 % de données en plus, comme passer de 84 000 à 105 000.
<!-- cellule -->
25 % de données en plus font gagner **0.0012 d'AUC, sur les 5 plis**. On réentraîne donc tout sur les 105 000 lignes, sans pouvoir mesurer directement ce que cela donne puisqu'il ne reste plus de lignes pour valider.
<!-- cellule -->
## 3. Réentraînement complet et fichiers de soumission

Les trois modèles × trois graines sont entraînés sur les 105 000 lignes. Le nombre d'itérations est la moyenne observée sur les cinq plis multipliée par 1.25, par analogie avec la règle `b / 0.85`. Les deux fichiers de soumission sont recalculés et comparés aux fichiers sauvegardés.
<!-- cellule -->
## 4. Soumissions et score public

Le score public porte sur environ 30 % du test (13 500 lignes). À 0.001 près, il dépend de ce tirage (voir le notebook 05). On choisit donc d'après la validation et la courbe d'apprentissage ; les scores publics ci-dessous vont dans le même sens.
<!-- cellule -->
**Soumission finale** : `submission_full_Bseeds.csv`, score public 0.92301. Le mélange à parts égales avec les modèles de plis, `submission_hybrid_50_50.csv`, fait 0.92259. Les deux sont copiés dans `submissions/`.
