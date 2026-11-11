# Soumissions

Le score public porte sur environ 30 % du test (13 500 lignes). Deux soumissions à moins de 0.001 d'écart ne sont pas départagées par ce score : nous choisissons d'après la validation sur cinq plis et la courbe d'apprentissage.

## Fichiers gardés

| Fichier | Contenu | Score public |
|---|---|---:|
| **`20261016_FINAL_full_Bseeds.csv`** | **soumission finale** : trois modèles × trois graines, réentraînés sur les 105 000 lignes | **0.92301** |
| `20261016_hybrid_50_50.csv` | moitié réentraînement complet, moitié modèles de plis | 0.92259 |
| `20261015_B_seeds_3_3_3_foldavg.csv` | trois modèles × trois graines, moyenne des modèles de plis (AUC sur cinq plis : 0.92073) | 0.92197 |

## Historique

| Date | Soumission | AUC, cinq plis | Score public |
|---|---|---:|---:|
| 2026-09-20 | CatBoost CPU profondeur 8, 56 colonnes | 0.91604 | 0.91857 |
| 2026-09-30 | le même sur GPU | 0.91498 | 0.91764 |
| 2026-10-14 | mélanges de CatBoost CPU, plusieurs graines (4 versions) | 0.91608 | 0.91846 à 0.91863 |
| 2026-10-15 | réglages × variables de la série 1 (4 versions) | 0.91816 à 0.91926 | 0.92067 à 0.92082 |
| 2026-10-15 | trois modèles, graine 42 | 0.92051 | 0.92190 |
| 2026-10-15 | trois graines + 10 % de LightGBM | 0.92073 | 0.92178 |
| 2026-10-15 | trois graines, moyenne des modèles de plis | 0.92073 | 0.92197 |
| 2026-10-16 | moitié réentraînement, moitié modèles de plis | — | 0.92259 |
| 2026-10-16 | **réentraînement sur les 105 000 lignes** | — | **0.92301** |
