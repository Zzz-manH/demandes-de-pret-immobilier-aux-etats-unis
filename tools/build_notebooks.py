"""Génère les six notebooks du projet.

    python tools/build_notebooks.py <dossier_sortie>

Le code de calcul est conservé dans tools/cells/*.py. Ce script assemble
la configuration, les explications, les résultats et les blocs de réentraînement.
Générer dans un autre dossier et vérifier avant de remplacer les notebooks existants.
"""
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

TOOLS = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else TOOLS.parent


def cell(name):
    return (TOOLS / 'cells' / f'{name}.py').read_text(encoding='utf-8').strip('\n')


def md(text=''):
    """Cellule de texte. Le contenu vient de tools/textes/<notebook>.md, dans l'ordre des cellules (voir save)."""
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip('\n'))


def raw_code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))


def naming_compatibility_cell():
    result = raw_code(cell('naming_compatibility'))
    result['metadata'].update({'jupyter': {'source_hidden': True}, 'tags': ['naming-compatibility']})
    return result


def gated(flag, body, gpu=True, catboost=True, setup=''):
    """Exécute ce bloc de calcul uniquement si son interrupteur est activé."""
    lines = [f'if {flag}:']
    if catboost:
        lines.append('    ensure_catboost()')
    if gpu:
        lines.append('    ensure_gpu()')
    if setup:
        lines.append(textwrap.indent(textwrap.dedent(setup).strip('\n'), '    '))
    lines.append(textwrap.indent(body.strip('\n'), '    '))
    lines.append('else:')
    lines.append(f'    print("Bloc désactivé ({flag} = False) ; affichage des résultats enregistrés dans artifacts/.")')
    return nbf.v4.new_code_cell('\n'.join(lines))


def save(cells, name):
    texts = (TOOLS / 'textes' / name.replace('.ipynb', '.md')).read_text(encoding='utf-8').split('<!-- cellule -->')
    markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']
    if len(texts) != len(markdown_cells):
        raise ValueError(f'{name} : {len(markdown_cells)} cellules de texte attendues, {len(texts)} dans tools/textes')
    for c, text in zip(markdown_cells, texts):
        c['source'] = text.strip().replace('{NOTEBOOK_MAP}', NOTEBOOK_MAP.strip()).replace('{TRAIN_NOTE}', TRAIN_NOTE.strip())
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'colab': {'provenance': []},
    }
    nbf.validate(nb)
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'{name}[{i}]', 'exec')
    print('wrote', OUT / name)


# ----------------------------------------------------------------------------- Cellules communes
SETUP = '''
# Environnement, chemins et fonctions partagés par les cinq notebooks
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = False


def find_project_dir():
    candidates = []
    if os.environ.get('HMDA_PROJECT_DIR'):
        candidates.append(Path(os.environ['HMDA_PROJECT_DIR']))
    here = Path.cwd().resolve()
    candidates += [here, *here.parents]
    candidates.append(Path('/content/drive/MyDrive/Colab Notebooks/demandes-de-pret-immobilier-aux-etats-unis'))
    for candidate in candidates:
        if (candidate / 'data' / 'train.csv').is_file() and (candidate / 'data' / 'test.csv').is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError('Projet introuvable (data/train.csv et data/test.csv requis) ; définir HMDA_PROJECT_DIR')


PROJECT_DIR = find_project_dir()
DATA_DIR = PROJECT_DIR / 'data'                                        # CSV sources (lecture seule)
ARTIFACTS = PROJECT_DIR / 'artifacts'                                  # Résultats dans un sous-dossier par notebook
PREPARED_DIR = ARTIFACTS / '01_data_exploration'                       # 01 : données préparées et schéma fixe
FEATURE_DIR = ARTIFACTS / '02_feature_engineering' / 'feature_store'   # 02 : feature store
SEED = 42
CATBOOST_VERSION = '1.2.10'
GPU_NAME = None
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})


def ensure_catboost():
    """Installe et importe la version fixée de CatBoost pour le réentraînement."""
    global catboost
    try:
        installed = importlib.metadata.version('catboost')
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != CATBOOST_VERSION:
        if 'catboost' in sys.modules:
            raise RuntimeError('Une autre version de CatBoost est déjà chargée ; redémarrer puis relancer le notebook')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', f'catboost=={CATBOOST_VERSION}'])
        importlib.invalidate_caches()
    import catboost as catboost_module
    if catboost_module.__version__ != CATBOOST_VERSION:
        raise RuntimeError(f'CatBoost {CATBOOST_VERSION} requis ; version chargée : {catboost_module.__version__}')
    catboost = catboost_module
    return catboost_module


def ensure_gpu():
    """Vérifie la présence du GPU NVIDIA avant le réentraînement."""
    global GPU_NAME
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('GPU requis : choisir un GPU T4 dans le type de session Colab, puis relancer')
    probe = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                           text=True, capture_output=True, check=False)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError('GPU NVIDIA indisponible : choisir un T4 puis relancer')
    GPU_NAME = probe.stdout.strip().splitlines()[0]
    return GPU_NAME


print('Projet :', PROJECT_DIR.name)
print(f'Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}')
'''

TRAIN_NOTE = '''
**Entraînement.** Tous les boutons de `TRAIN` sont à `False` : les notebooks relisent les résultats sauvegardés dans `artifacts/`. Deux exceptions : le notebook 01 entraîne un petit modèle de contrôle sur CPU et le notebook 02 réécrit les variables. Pour refaire une expérience, mettre son bouton à `True` et choisir un **GPU T4** si le bloc le demande. Les modèles déjà entraînés sont sauvegardés et ne sont pas refaits.
'''

DATA = cell('data_contract') + '\n\n' + cell('helpers')

FEATURE_LOADER = '''
# Charge les features produites par 02, nécessaires au réentraînement.
def load_feature_store():
    meta_path = FEATURE_DIR / 'feature_store.json'
    require(meta_path.is_file(), 'Feature store absent : exécuter 02_feature_engineering.ipynb')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    require(meta['data_hashes'] == DATA_HASHES, 'Feature store incompatible avec les données de 01 : relancer 02')
    stores = {}
    for part, expected_rows in [('train', len(X_base)), ('test', len(X_test_base))]:
        path = FEATURE_DIR / f'blocks_{part}.parquet'
        require(sha256_file(path) == meta['files'][path.name], f'SHA-256 incorrect pour {path.name} : relancer 02')
        frame = pd.read_parquet(path)
        require(len(frame) == expected_rows, f'Nombre de lignes incorrect : {path.name}')
        for column in frame.columns:
            frame[column] = frame[column].astype(object if column in meta['categorical'] else 'float64')
        stores[part] = frame
    blocks = {name: (stores['train'][cols].reset_index(drop=True), stores['test'][cols].reset_index(drop=True))
              for name, cols in meta['blocks'].items()}
    return blocks, set(meta['categorical']), meta


def variant_frames(blocks):
    """Ajoute les blocs choisis aux 56 colonnes sources, comme dans 02."""
    train_parts = [X_base] + [BLOCKS[b][0] for b in blocks]
    test_parts = [X_test_base] + [BLOCKS[b][1] for b in blocks]
    train = pd.concat(train_parts, axis=1)
    test = pd.concat(test_parts, axis=1)
    require(train.columns.is_unique and train.columns.equals(test.columns), 'Colonnes incohérentes entre les variantes train/test')
    require(train.iloc[:, :len(FEATURE_COLUMNS)].equals(X_base), 'Les 56 colonnes sources ont été modifiées')
    cats = CATEGORICAL_COLUMNS + [c for c in train.columns if c in NEW_CATEGORICAL]
    return train, test, cats


if (FEATURE_DIR / 'feature_store.json').is_file():
    BLOCKS, NEW_CATEGORICAL, FEATURE_META = load_feature_store()
    print(f"Feature store : {len(BLOCKS)} blocs, {sum(len(c) for c in FEATURE_META['blocks'].values())} colonnes ; "
          f"jeux de features {[semantic_label(name) for name in FEATURE_META['feature_sets']]}")
else:
    BLOCKS, NEW_CATEGORICAL, FEATURE_META = None, set(), None
    print('Feature store absent : lecture des résultats possible ; exécuter 02 avant le réentraînement.')
'''

RESULT_HELPERS = '''
# Lit les résultats JSON par modèle et fold pour des comparaisons appariées.
def load_runs(directory, pattern='*__fold?.json'):
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    require(paths, f'Aucun résultat dans {directory} : synchroniser artifacts depuis Drive ou activer le réentraînement')
    return pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in paths])


def paired_table(runs, key, baseline, min_gain=0.0005, min_folds=4):
    auc = runs.pivot(index='fold', columns=key, values='valid_auc')
    require(baseline in auc.columns, f'Référence absente : {baseline}')
    delta = auc.sub(auc[baseline], axis=0)
    table = pd.DataFrame({
        'folds': auc.notna().sum(),
        'mean_auc': auc.mean(),
        'delta_vs_base': delta.mean(),
        'folds_better': (delta > 0).sum(),
    })
    table['keep'] = ((table['folds'] == 5) & (table['delta_vs_base'] >= min_gain)
                     & (table['folds_better'] >= min_folds))
    table.loc[baseline, 'keep'] = False
    return table.sort_values('mean_auc', ascending=False), delta


def plot_deltas(delta, baseline, title):
    others = delta.drop(columns=[baseline])
    order = others.mean().sort_values().index.tolist()
    means = others[order].mean() * 1e4
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.6))
    ax.barh(range(len(order)), means, color=['#4c72b0' if v > 0 else '#c44e52' for v in means])
    for i, name in enumerate(order):
        ax.scatter(others[name] * 1e4, np.full(len(others), i), color='black', s=12, zorder=3)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(range(len(order)), [semantic_label(name) for name in order])
    ax.set_xlabel(f'Écart AUC / {semantic_label(baseline)} (×1e-4) ; barre = moyenne, points = folds')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
'''

OOF_HELPERS = '''
# Lit les prédictions OOF des cinq folds externes.
def read_oof(directory, group):
    frames = [pd.read_csv(Path(directory) / f'{group}__fold{k}__valid.csv').assign(fold=k) for k in range(1, 6)]
    return pd.concat(frames).sort_values('id', kind='stable').reset_index(drop=True)


def fold_auc(frame, probability):
    probability = np.asarray(probability)
    return pd.Series({k: roc_auc_score(frame.loc[frame['fold'].eq(k), 'accepted'],
                                       probability[frame['fold'].eq(k).to_numpy()]) for k in range(1, 6)})


def mean_test(paths):
    frames = [pd.read_csv(p, dtype={'id': str}) for p in paths]
    for f in frames:
        require(f['id'].equals(sample_submission['id']), f'Ordre des ID de test incohérent : {Path(paths[0]).name}')
    return np.mean([f['accepted'].to_numpy() for f in frames], axis=0)


CV_DIR = ARTIFACTS / '04_final_model' / 'cv_models'      # Modèles sur cinq folds (la référence initiale et les trois modèles finaux × 3 seeds)
FULL_DIR = ARTIFACTS / '04_final_model' / 'full_refit'   # Courbe d’apprentissage et modèles sur toutes les données
FINAL_MODEL_NAMES = [MODEL_DEPTH6_ENRICHED, MODEL_DEPTH8_ENRICHED, MODEL_DEPTH6_REDUCED]
SEED_TAGS = {42: '', 2027: '_s2027', 31415: '_s31415'}
'''

NOTEBOOK_MAP = '''
Les cinq notebooks : 01 données → 02 variables → 03 choix des modèles → 04 modèle final → 05 analyse des erreurs. Les CSV sont dans `data/`, les résultats dans `artifacts/<nom du notebook>/`.
'''

# ============================================================================= 01
nb01 = [
    md(''),
    code(SETUP),
    naming_compatibility_cell(),
    code('''
import sklearn

RANDOM_STATE = 42
N_SPLITS = 5
SCHEMA_VERSION = "1.0"
MISSING_CATEGORY = "MISSING"
REBUILD_PREPARED = False   # True : réécrit les données préparées ; les nouvelles empreintes empêchent la reprise des anciens résultats
pd.set_option("display.max_colwidth", 75)
print(f"Données sources : {DATA_DIR.relative_to(PROJECT_DIR)}\\nDonnées préparées : {PREPARED_DIR.relative_to(PROJECT_DIR)}\\nsklearn {sklearn.__version__}")
'''),
    md(''),
    raw_code(cell('prep_read')),
    md(''),
    raw_code(cell('prep_types')),
    md(''),
    raw_code(cell('prep_normalize')),
    md(''),
    raw_code(cell('prep_verify')),
    md(''),
    raw_code(cell('prep_overview')),
    md(''),
    code('''
overall_rate = y.mean()
ltv = X_train['loan_to_value_ratio']
ltv_bins = pd.cut(ltv, [-np.inf, 60, 80, 90, 95, 97, 100, np.inf])
ltv_band = ltv_bins.astype(str).where(ltv.notna(), 'missing')
ltv_order = [str(c) for c in ltv_bins.cat.categories] + ['missing']
dti_order = ['<20%', '20%-<30%', '30%-<36%', *[str(v) for v in range(36, 50)], '50%-60%', '>60%', MISSING_CATEGORY]


def rate_bar(ax, values, title, order=None, top=12):
    table = pd.DataFrame({'v': np.asarray(values), 'y': y.to_numpy()}).groupby('v')['y'].agg(['mean', 'size'])
    table = table.reindex([v for v in order if v in table.index]) if order else \\
        table.sort_values('size', ascending=False).head(top)
    ax.bar(range(len(table)), table['mean'], color='#4c72b0')
    ax.axhline(overall_rate, color='black', ls='--', lw=1)
    ax.set_xticks(range(len(table)), [str(v) for v in table.index], rotation=60, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel('taux d’acceptation')


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
rate_bar(axes[0, 0], X_train['loan_type'], 'loan_type (1=classique, 2=FHA, 3=VA, 4=RHS)')
rate_bar(axes[0, 1], X_train['aus-1'], 'aus-1 (système automatisé ; 6 = non utilisé)')
rate_bar(axes[0, 2], X_train['debt_to_income_ratio'], 'debt_to_income_ratio', order=dti_order)
rate_bar(axes[1, 0], ltv_band, 'tranche de loan_to_value_ratio', order=ltv_order)
rate_bar(axes[1, 1], X_train['applicant_credit_score_type'], 'applicant_credit_score_type')
rate_bar(axes[1, 2], X_train['occupancy_type'], 'occupancy_type (1=résidence principale)')
plt.tight_layout()
plt.show()
print(f'Taux d’acceptation global : {overall_rate:.3f}')
'''),
    md(''),
    code('''
missing_signal = []
for column in ['loan_to_value_ratio', 'property_value', 'income', 'loan_term']:
    mask = X_train[column].isna()
    missing_signal.append({'Variable': column, 'Lignes manquantes': int(mask.sum()),
                           'Acceptation si absente': y[mask].mean(), 'Acceptation si présente': y[~mask].mean()})
for column in ['debt_to_income_ratio', 'applicant_age']:
    mask = X_train[column].eq(MISSING_CATEGORY)
    missing_signal.append({'Variable': column, 'Lignes manquantes': int(mask.sum()),
                           'Acceptation si absente': y[mask].mean(), 'Acceptation si présente': y[~mask].mean()})
display_named(pd.DataFrame(missing_signal).round(3))

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, (column, use_log, clip) in zip(axes, [('loan_amount', True, None), ('income', True, None),
                                                ('property_value', True, None), ('loan_to_value_ratio', False, (0, 150))]):
    for label, colour, name in [(1, '#4c72b0', 'acceptée'), (0, '#dd8452', 'refusée')]:
        values = X_train.loc[y.eq(label) & X_train[column].notna(), column]
        if use_log:
            values = np.log10(values[values > 0])
        if clip:
            values = values.clip(*clip)
        ax.hist(values, bins=60, density=True, alpha=0.5, color=colour, label=name)
    ax.set_title(column + (' (log10)' if use_log else ', borné à 0–150'))
axes[0].legend()
plt.tight_layout()
plt.show()
'''),
    md(''),
    raw_code(cell('prep_audit')),
    md(''),
    code('''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

combined = pd.concat([X_train, X_test], ignore_index=True)
encoded = combined.copy()
for column in categorical_columns:
    encoded[column] = encoded[column].astype('category').cat.codes
is_test = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
adversarial_auc = cross_val_score(
    HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE), encoded, is_test,
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc').mean()

train_id_values = pd.to_numeric(train_labels['id'])
test_id_values = pd.to_numeric(test_ids['id'])
sorted_labels = y.to_numpy()[np.argsort(train_id_values.to_numpy())].astype(float)
window = 5
neighbour_sum = pd.Series(sorted_labels).rolling(2 * window + 1, center=True, min_periods=1).sum() - sorted_labels
neighbour_count = pd.Series(np.ones(len(sorted_labels))).rolling(2 * window + 1, center=True, min_periods=1).sum() - 1
neighbour_corr = np.corrcoef(sorted_labels, neighbour_sum / neighbour_count)[0, 1]

loan_type_name = X_train['loan_type'].map({'1': 'Conventional', '2': 'FHA', '3': 'VA', '4': 'FSA/RHS'})
lien_name = X_train['lien_status'].map({'1': 'First Lien', '2': 'Subordinate Lien'})
construction_name = X_train['construction_method'].map({'1': 'Site-Built', '2': 'Manufactured'})
checks = pd.DataFrame([
    ('AUC adversariale train/test (0,5 = absence de séparation détectée)', f'{adversarial_auc:.4f}'),
    ('AUC de la cible prédite par l’ID seul', f'{roc_auc_score(y, train_id_values):.4f}'),
    ('Corrélation de la cible avec la moyenne des 10 ID voisins', f'{neighbour_corr:.4f}'),
    ('Plage des ID : train / test', f'{train_id_values.min()}–{train_id_values.max()} / '
                           f'{test_id_values.min()}–{test_id_values.max()}'),
    ('Produit = type de prêt + rang de garantie', f"{(loan_type_name + ':' + lien_name).eq(X_train['derived_loan_product_type']).mean():.2%}"),
    ('Cohérence entre catégorie de logement et construction',
     f"{X_train['derived_dwelling_category'].str.split(':').str[1].eq(construction_name).mean():.2%}"),
], columns=['Contrôle', 'Résultat'])
display_named(checks)

loan_to_value_raw = X_train['loan_amount'] / X_train['property_value']
anomalies = {
    'Ensemble': pd.Series(True, index=X_train.index),
    'LTV absent': ltv.isna(),
    'LTV > 150': ltv > 150,
    'Population du secteur = 0': X_train['tract_population'].eq(0),
    'Revenu régional = 0': X_train['ffiec_msa_md_median_family_income'].eq(0),
    'Montant du prêt / valeur du bien > 2': loan_to_value_raw > 2,
    'Revenu ≤ 0': X_train['income'] <= 0,
}
display_named(pd.DataFrame([{'Cas': name, 'Lignes': int(mask.sum()), 'Taux d’acceptation': y[mask].mean()}
                      for name, mask in anomalies.items()]).round(3))
'''),
    md(''),
    md(''),
    raw_code(cell('prep_folds')),
    md(''),
    raw_code(cell('prep_write')),
    md(''),
    raw_code(cell('prep_readback')),
]

# ============================================================================= 02
ROUND1_TABLE = '''
| Bloc | Colonnes ajoutées | Idée |
|---|---|---|
| `dti` | `dti_lower`, `dti_upper` | Bornes numériques des catégories de DTI (ex. `30%-<36%`) |
| `ltv` | `ltv_calc`, `ltv_gap` | LTV recalculé à partir du prêt et de la valeur du bien ; écart au LTV déclaré |
| `income` | `income_to_area`, `loan_to_income`, `value_to_income` | Revenu relatif à la région ; prêt et valeur du bien relatifs au revenu |
| `missing` | `missing_pattern` (catégorie), `missing_count` | Combinaisons de valeurs manquantes |
| `freq` | `freq_county_code`, `freq_derived_msa-md`, `freq_state_code` | Fréquence géographique dans train + test, sans cible |
'''

ROUND2_TABLE = '''
| Bloc | Colonnes ajoutées | Motivation |
|---|---|---|
| `precision` | `ltv_precision` (absent / entier / 1–3 décimales / plus) | Acceptation : 90,7 % pour un LTV entier contre 84,4 % à deux décimales ; possible signal des pratiques de déclaration |
| `thresholds` | `ltv_band` (catégorie), `ltv_over_program` | Seuils testés : 80, 95, 97, 100 ; écart au plafond de référence (classique 97, FHA 96,5, VA/RHS 100) |
| `implied` | `implied_property_value`, `implied_value_ratio` | Valeur du bien déduite du prêt et du LTV, comparée à la valeur déclarée |
| `term` | `term_is_360`, `term_years` | Durée en années et indicateur de prêt sur 30 ans |
'''

ROUND3_TABLE = '''
| Bloc | Colonnes ajoutées | Motivation |
|---|---|---|
| `purpose` | `product_flags` (7 indicateurs regroupés) | Dispersion résiduelle la plus élevée, sur 17 groupes seulement |
| `ltv_consistency` | `ltv_consistency`, `ltv_outside_interval` | Intervalle de LTV compatible avec les montants arrondis par tranches de 10 000 dollars ; écart éventuel de définition |
| `risk_layers` | `risk_layer_count`, `risk_layer_pattern` | Cumul : LTV/DTI élevés, sans AUS, résidence non principale, LTV absent, garantie de second rang |
| `tract_income` | `income_to_tract` | Revenu relatif à la médiane du secteur de recensement |
'''

STORE_SAVE = '''
# Sauvegarde les colonnes ajoutées ; les 56 colonnes sources restent fournies par 01.
FEATURE_SETS = {
    FEATURE_SET_INITIAL: ['dti', 'ltv', 'income', 'missing', 'freq'],                                   # Tous les blocs de la série 1 (68 colonnes)
    FEATURE_SET_ENRICHED: ['dti', 'ltv', 'income', 'missing', 'precision', 'thresholds', 'implied', 'term'],      # les deux modèles enrichis finaux (72 colonnes)
    FEATURE_SET_REDUCED: ['dti', 'ltv', 'income', 'missing', 'precision', 'thresholds'],                      # CatBoost profondeur 6 réduit final (68 colonnes)
}
block_columns = {name: list(train_block.columns) for name, (train_block, _) in BLOCKS.items()}
all_columns = [c for columns in block_columns.values() for c in columns]
require(len(all_columns) == len(set(all_columns)), 'Noms de colonnes dupliqués entre blocs')
train_store = pd.concat([BLOCKS[name][0] for name in block_columns], axis=1)
test_store = pd.concat([BLOCKS[name][1] for name in block_columns], axis=1)
require(train_store.shape == (len(X_base), len(all_columns))
        and test_store.shape == (len(X_test_base), len(all_columns)), 'Dimensions du feature store incorrectes')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
store_paths = {'train': FEATURE_DIR / 'blocks_train.parquet', 'test': FEATURE_DIR / 'blocks_test.parquet'}
train_store.to_parquet(store_paths['train'], index=False)
test_store.to_parquet(store_paths['test'], index=False)
write_json(FEATURE_DIR / 'feature_store.json', {
    'blocks': block_columns,
    'categorical': sorted(c for c in NEW_CATEGORICAL if c in all_columns),
    'feature_sets': FEATURE_SETS,
    'rows': {'train': len(train_store), 'test': len(test_store)},
    'data_hashes': DATA_HASHES,
    'files': {path.name: sha256_file(path) for path in store_paths.values()},
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
})
for part, frame in [('train', train_store), ('test', test_store)]:
    restored = pd.read_parquet(store_paths[part])
    pd.testing.assert_frame_equal(restored.reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False)
display_named(pd.DataFrame([{'Bloc': name, 'Nombre de colonnes': len(cols), 'Colonnes': ', '.join(cols)} for name, cols in block_columns.items()]))
print(f'Variables enregistrées dans {FEATURE_DIR.relative_to(PROJECT_DIR)} : {len(all_columns)} colonnes, relecture vérifiée.')
'''

def ablation_results(directory, baseline, min_gain, title):
    return f'''
runs = load_runs(ARTIFACTS / '02_feature_engineering' / '{directory}')
table, delta = paired_table(runs, 'variant', {baseline}, min_gain={min_gain})
table['feature_count'] = runs.groupby('variant')['feature_count'].first()
display_named(table.round(6))
plot_deltas(delta, {baseline}, '{title}')
'''

nb02 = [
    md(''),
    code(SETUP),
    naming_compatibility_cell(),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'ablation_round1': False,   # Série 1 : 7 variantes × 5 folds, environ 70 min
    'ablation_round2': False,   # Série 2 : 8 variantes × 5 folds, environ 80 min
    'ablation_round3': False,   # Série 3 : 6 variantes × 5 folds, environ 60 min
}
'''),
    md(''),
    raw_code(DATA),
    md(''),
    raw_code(cell('dti_bounds')),
    md(f'### Blocs de la première série\n\nUn ratio exige un dénominateur strictement positif ; sinon il est manquant, sans ajout arbitraire d’epsilon.\n\n{ROUND1_TABLE}'),
    raw_code(cell('features_round1')),
    md(f'## 3. Features de la deuxième série\n\n`ltv_precision` utilise le **texte source** du LTV (ex. `80.0` ou `79.85`) : cette colonne est donc relue en chaîne.\n\n{ROUND2_TABLE}'),
    raw_code(cell('features_round2')),
    md(f'## 4. Features de la troisième série\n\n{ROUND3_TABLE}'),
    raw_code(cell('features_round3')),
    md(''),
    raw_code(STORE_SAVE),
    md(''),
    raw_code(RESULT_HELPERS),
    md(''),
    raw_code(cell('ablation_r1_config')),
    gated("TRAIN['ablation_round1']", cell('ablation_run') + '\n\n' + cell('ablation_r1_summary')),
    code(ablation_results('ablation_round1', 'ABLATION_ORIGINAL', 0.0003, 'Ablation des features, série 1')),
    md(''),
    md(''),
    raw_code(cell('ablation_r2_config')),
    gated("TRAIN['ablation_round2']", cell('ablation_run') + '\n\n' + cell('ablation_r2_summary')),
    code(ablation_results('ablation_round2', 'ABLATION_INITIAL_REFERENCE', 0.0003, 'Ablation des features, série 2')),
    md(''),
    md(''),
    raw_code(cell('ablation_r3_config')),
    gated("TRAIN['ablation_round3']", cell('ablation_run') + '\n\n' + cell('ablation_r3_summary')),
    code(ablation_results('ablation_round3', 'ABLATION_ENRICHED_REFERENCE', 0.0005, 'Ablation des features, série 3')),
    md(''),
]

# ============================================================================= 03
BASELINE_RESULTS = '''
cpu_run = ARTIFACTS / '03_model_selection' / 'baseline_catboost_xgboost'   # CatBoost de profondeur 6/8, XGBoost (CPU)
lr_run = ARTIFACTS / '03_model_selection' / 'baseline_logistic'            # Régression logistique
rows = []
for path in [cpu_run / 'summary.csv', lr_run / 'summary.csv', cpu_run / 'blend_summary.csv']:
    for r in pd.read_csv(path).itertuples():
        rows.append({'Modèle': r.model, 'AUC sur cinq folds': r.mean_fold_auc, 'Écart-type': r.std_fold_auc,
                     'OOF LogLoss': r.pooled_oof_logloss})
rows.append({'Modèle': 'CatBoost profondeur 8 (GPU T4, mêmes paramètres)', 'AUC sur cinq folds': 0.91498, 'Écart-type': np.nan, 'OOF LogLoss': np.nan})
display_named(pd.DataFrame(rows).sort_values('AUC sur cinq folds', ascending=False).reset_index(drop=True).round(5))
'''

BASELINE_RERUN = '''
# Réentraîne CatBoost de profondeur 6/8 sur CPU (1600 itérations) dans un dossier distinct.
RERUN_DIR = ARTIFACTS / '03_model_selection' / 'baseline_catboost_rerun'
RERUN_DIR.mkdir(parents=True, exist_ok=True)
base_params = dict(iterations=1600, learning_rate=0.05, l2_leaf_reg=3.0, loss_function='Logloss',
                   eval_metric='AUC', nan_mode='Min', random_seed=SEED, task_type='CPU',
                   thread_count=os.cpu_count() or 2, use_best_model=False,
                   allow_writing_files=False, verbose=False)
rerun_rows = []
for fold in range(1, 6):
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    for name, depth in [(BASELINE_CATBOOST_DEPTH6, 6), (BASELINE_CATBOOST_DEPTH8, 8)]:
        path = RERUN_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            rerun_rows.append(json.loads(path.read_text(encoding='utf-8')))
            continue
        started = time.perf_counter()
        model = catboost.CatBoostClassifier(**base_params, depth=depth)
        model.fit(X_base.iloc[fit_idx], y.iloc[fit_idx], cat_features=CATEGORICAL_COLUMNS)
        probability = checked_probabilities(model, X_base.iloc[valid_idx])
        row = {'model': name, 'fold': fold, 'valid_auc': roc_auc_score(y.iloc[valid_idx], probability),
               'seconds': time.perf_counter() - started}
        write_json(path, row)
        rerun_rows.append(row)
        print(f"{name} fold {fold}: AUC={row['valid_auc']:.6f}  {row['seconds'] / 60:.1f} min", flush=True)
display_named(pd.DataFrame(rerun_rows).pivot(index='fold', columns='model', values='valid_auc').agg(['mean', 'std']).round(6))
'''

TRANSFORM_RESULTS = '''
TRANSFORM_DIR = ARTIFACTS / '03_model_selection' / 'transformations'
transform = pd.read_csv(TRANSFORM_DIR / 'auc_summary.csv')
order = ['base', 'poly2', 'poly3', 'quantile', 'rbf', 'spline']
display_named(transform.pivot(index='representation', columns='model', values='auc_mean').reindex(order).round(5))
if (TRANSFORM_DIR / 'auc_comparison.png').is_file():
    display_named(Image(filename=str(TRANSFORM_DIR / 'auc_comparison.png'), width=760))
'''

TUNING_RESULTS = '''
TUNING_DIR = ARTIFACTS / '03_model_selection' / 'catboost_tuning'
tuning_runs = load_runs(TUNING_DIR)
fold1 = tuning_runs[tuning_runs['fold'].eq(1)].set_index('candidate')
screen = pd.DataFrame({
    'AUC fold 1': fold1['valid_auc'],
    'Écart à la référence à 1 600 itérations': fold1['valid_auc'] - fold1.loc[TUNING_DEPTH8_FIXED1600, 'valid_auc'],
    'Itérations de réentraînement': fold1['refit_rounds'],
}).sort_values('AUC fold 1', ascending=False)
print('Étape 1 : sélection sur le fold 1')
display_named(screen.round(6))
finalists = [TUNING_DEPTH8_FIXED1600, TUNING_DEPTH8_BERNOULLI80, TUNING_DEPTH8_L2_30_BERNOULLI80, TUNING_DEPTH6_BERNOULLI80]
table, delta = paired_table(tuning_runs[tuning_runs['candidate'].isin(finalists)], 'candidate', TUNING_DEPTH8_FIXED1600)
print('Étape 2 : confirmation des finalistes sur cinq folds')
display_named(table.round(6))
plot_deltas(delta, TUNING_DEPTH8_FIXED1600, 'Réglage CatBoost, cinq folds (référence : CatBoost profondeur 8 GPU)')
'''

SCREEN_RESULTS = '''
screen_runs = load_runs(ARTIFACTS / '03_model_selection' / 'training_modes')
table, delta = paired_table(screen_runs, 'variant', TRAINING_STANDARD)
display_named(table.round(6))
plot_deltas(delta, TRAINING_STANDARD, 'Modes de calcul CatBoost, cinq folds')
'''

nb03 = [
    md(''),
    code(SETUP),
    naming_compatibility_cell(),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'baseline_cpu': False,   # Baselines CatBoost de profondeur 6/8 sur CPU (environ 3 h sur Colab)
    'tuning': False,         # Réglage CatBoost sur cinq folds (GPU, environ 2,5 h)
    'screen': False,         # Sélection des modes de calcul (GPU, environ 2 h)
}
'''),
    md(''),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS),
    md(''),
    code(BASELINE_RESULTS),
    gated("TRAIN['baseline_cpu']", BASELINE_RERUN, gpu=False),
    md(''),
    md(''),
    code(TRANSFORM_RESULTS),
    md(''),
    raw_code(cell('tuning_config')),
    gated("TRAIN['tuning']", cell('tuning_run') + '\n\n' + cell('tuning_summary'),
          setup="X_dti = pd.concat([X_base, BLOCKS['dti'][0]], axis=1)   # Bornes DTI utilisées par le candidat avec bornes DTI"),
    code(TUNING_RESULTS),
    md(''),
    md(''),
    raw_code(cell('screen_config')),
    gated("TRAIN['screen']", cell('screen_preflight') + '\n\n' + cell('screen_run') + '\n\n' + cell('screen_summary')),
    code(SCREEN_RESULTS),
    md(''),
    md(''),
]

# ============================================================================= 04
CV_RESULTS = '''
cv_runs = load_runs(CV_DIR)
per_group = cv_runs.pivot(index='fold', columns='group', values='valid_auc')
display_named(per_group.agg(['mean', 'std']).T.sort_values('mean', ascending=False).round(6))

oof = {g: read_oof(CV_DIR, g) for g in [MODEL_DEPTH6_INITIAL] + [b + t for b in FINAL_MODEL_NAMES for t in SEED_TAGS.values()]}
reference = oof[MODEL_DEPTH6_INITIAL]
for name, frame in oof.items():
    require(frame['id'].equals(reference['id']) and frame['fold'].equals(reference['fold']), f'Prédictions OOF non alignées : {name}')
blend_seed42 = np.mean([oof[b]['prob'] for b in FINAL_MODEL_NAMES], axis=0)
multiseed_ensemble = np.mean([np.mean([oof[b + t]['prob'] for t in SEED_TAGS.values()], axis=0) for b in FINAL_MODEL_NAMES], axis=0)
candidates = {
    'Référence initiale : profondeur 6, 68 variables': reference['prob'],
    'CatBoost profondeur 6, variables réduites': oof[MODEL_DEPTH6_REDUCED]['prob'],
    'ensemble des trois modèles finaux (seed 42)': blend_seed42,
    'Ensemble final : trois configurations × trois graines': multiseed_ensemble,
}
base_folds = fold_auc(reference, reference['prob'])
fold_table = pd.DataFrame({name: fold_auc(reference, prob) for name, prob in candidates.items()})
summary = pd.DataFrame({'AUC sur cinq folds': fold_table.mean(), 'Écart à la référence initiale': fold_table.sub(base_folds, axis=0).mean(),
                        'Folds améliorés': fold_table.sub(base_folds, axis=0).gt(0).sum()})
display_named(summary.round(6))

fig, ax = plt.subplots(figsize=(7, 3.8))
for name in fold_table.columns:
    ax.plot(fold_table.index, fold_table[name], marker='o', label=name)
ax.set_xticks(range(1, 6))
ax.set_xlabel('Fold')
ax.set_ylabel('AUC de validation')
ax.set_title('Modèles finaux sur cinq folds')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
'''

LC_RESULTS = '''
lc = load_runs(FULL_DIR, LEARNING_CURVE_RUN_PATTERN)
curve = lc.pivot(index='fold', columns='name', values='valid_auc')
curve['gain_84k_vs_67k'] = curve[LEARNING_CURVE_FULL] - curve[LEARNING_CURVE_SUBSET]
display_named(curve.round(6))
print(f"Données ×1.25 (67 200 → 84 000 lignes) : gain AUC moyen {curve['gain_84k_vs_67k'].mean():+.5f}, "
      f"{int((curve['gain_84k_vs_67k'] > 0).sum())}/5 folds positifs")

fig, ax = plt.subplots(figsize=(6, 3.8))
for fold, row in curve.iterrows():
    ax.plot([67200, 84000], [row[LEARNING_CURVE_SUBSET], row[LEARNING_CURVE_FULL]], marker='o', label=f'fold {fold}')
ax.set_xlabel('Lignes d’entraînement')
ax.set_ylabel('AUC de validation')
ax.set_title('Courbe d’apprentissage : CatBoost profondeur 6 réduit, 80 % et 100 % du train externe')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
'''

FULL_RESULTS = '''
full_runs = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in sorted(FULL_DIR.glob('FULL_*.json'))])
display_named(full_runs[['name', 'base', 'seed', 'train_rows', 'rounds']].sort_values(['base', 'seed']))
full_pred = np.mean([mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for s in SEED_TAGS]) for b in FINAL_MODEL_NAMES], axis=0)
fold_pred = np.mean([mean_test([CV_DIR / f'{b}{t}__fold{k}__test.csv' for t in SEED_TAGS.values() for k in range(1, 6)])
                     for b in FINAL_MODEL_NAMES], axis=0)
checks = []
for label, path, expected in [
    ('Soumission finale l’ensemble complet multigraines', FULL_DIR / FULL_ENSEMBLE_SUBMISSION, full_pred),
    ('Alternative hybrid_50_50', FULL_DIR / 'submission_hybrid_50_50.csv', 0.5 * full_pred + 0.5 * fold_pred),
]:
    saved = pd.read_csv(path, dtype={'id': str})
    require(saved['id'].equals(sample_submission['id']), f'Ordre des ID différent du modèle de soumission : {path.name}')
    checks.append({'Fichier': label, 'Écart maximal au recalcul': float(np.abs(saved['accepted'].to_numpy() - expected).max())})
display_named(pd.DataFrame(checks))
print(f'Corrélation des prédictions test, modèles complets / moyenne des folds : {np.corrcoef(full_pred, fold_pred)[0, 1]:.5f}')
'''

PUBLIC_TABLE = '''
public = pd.DataFrame([
    ('CatBoost profondeur 8 CPU, 56 features sources', 0.91604, 0.91857),
    ('CatBoost profondeur 8 GPU (mêmes paramètres)', 0.91498, 0.91764),
    ('CatBoost profondeur 6 + features de la série 1 (la référence initiale)', 0.91931, 0.92073),
    ('ensemble des trois modèles finaux (seed 42), moyenne des folds', 0.92051, 0.92190),
    ('Ensemble final : trois configurations × trois graines, moyenne des folds', 0.92073, 0.92197),
    ('hybrid : 50 % données complètes + 50 % moyenne des folds', np.nan, 0.92259),
    ('l’ensemble complet multigraines : 9 modèles sur les 105 000 lignes (FINAL)', np.nan, 0.92301),
], columns=['Soumission', 'AUC sur cinq folds', 'AUC publique'])
display_named(public)
'''

nb04 = [
    md(''),
    code(SETUP),
    naming_compatibility_cell(),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'cv_models': False,        # la référence initiale et les trois modèles finaux × 3 seeds, cinq folds (GPU, environ 6 h)
    'learning_curve': False,   # Courbe d’apprentissage : 10 entraînements (GPU, environ 75 min)
    'full_refit': False,       # Réentraînement complet : 9 modèles (GPU, environ 50 min)
}
'''),
    md(''),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS + '\n\n' + OOF_HELPERS),
    md(''),
    raw_code(cell('cv_config')),
    gated("TRAIN['cv_models']", cell('cv_run') + '\n\n' + cell('cv_blend')),
    code(CV_RESULTS),
    md(''),
    md(''),
    raw_code(cell('refit_config')),
    gated("TRAIN['learning_curve'] or TRAIN['full_refit']", cell('refit_functions')),
    gated("TRAIN['learning_curve']", cell('refit_learning_curve')),
    code(LC_RESULTS),
    md(''),
    md(''),
    gated("TRAIN['full_refit']", cell('refit_full') + '\n\n' + cell('refit_submit')),
    code(FULL_RESULTS),
    md(''),
    code(PUBLIC_TABLE),
    md(''),
]

# ============================================================================= 05
RESIDUAL = '''
reduced_model_oof = read_oof(CV_DIR, MODEL_DEPTH6_REDUCED)
diag = X_base.assign(id=train_ids.astype('int64').to_numpy(), accepted=y.to_numpy()).merge(
    reduced_model_oof[['id', 'prob']], on='id', validate='one_to_one')
residual = diag['accepted'] - diag['prob']
variance = diag['prob'] * (1 - diag['prob'])


def dispersion(groups, min_rows):
    t = pd.DataFrame({'g': np.asarray(groups).astype(str), 'r': residual.to_numpy(), 'v': variance.to_numpy()})
    t = t.groupby('g').agg(r=('r', 'sum'), v=('v', 'sum'), n=('r', 'size'))
    t = t[t['n'] >= min_rows]
    return float((t['r'] ** 2 / t['v']).mean()), len(t)


ltv = diag['loan_to_value_ratio']
product_flags = ['business_or_commercial_purpose', 'open-end_line_of_credit', 'reverse_mortgage',
                 'interest_only_payment', 'balloon_payment', 'negative_amortization', 'other_nonamortizing_features']
tract_columns = ['county_code', 'tract_population', 'tract_minority_population_percent',
                 'tract_to_msa_income_percentage', 'tract_owner_occupied_units']
rng = np.random.default_rng(SEED)
groupings = [
    ('Secteur de recensement', diag[tract_columns].astype(str).agg('|'.join, axis=1), 5),
    ('Comté', diag['county_code'], 5),
    ('MSA', diag['derived_msa-md'], 5),
    ('État', diag['state_code'], 5),
    ('Profil institutionnel : AUS × dépôt × bénéficiaire × type de score × préapprobation',
     diag[['aus-1', 'submission_of_application', 'initially_payable_to_institution',
           'applicant_credit_score_type', 'preapproval']].astype(str).agg('|'.join, axis=1), 5),
    ('Groupes aléatoires (référence, environ 8 000)', rng.integers(0, 8000, len(diag)), 5),
    ('Type de prêt × DTI', diag['loan_type'] + '|' + diag['debt_to_income_ratio'], 30),
    ('Tranche de LTV × DTI', pd.cut(ltv, [0, 60, 80, 90, 95, 97, 100, np.inf]).astype(str) + '|' + diag['debt_to_income_ratio'], 30),
    ('Âge de l’emprunteur × âge du coemprunteur', diag['applicant_age'] + '|' + diag['co-applicant_age'], 30),
    ('Combinaison des indicateurs de produit', diag[product_flags].astype(str).agg('|'.join, axis=1), 30),
    ('Groupes aléatoires (référence, 300)', rng.integers(0, 300, len(diag)), 30),
]
display_named(pd.DataFrame([{'Regroupement': name, 'Dispersion résiduelle normalisée': dispersion(groups, min_rows)[0],
                       'Groupes': dispersion(groups, min_rows)[1], 'Effectif minimal': min_rows}
                      for name, groups, min_rows in groupings]).round(3))
'''

ANOMALY = '''
loan_value = diag['loan_amount'] / diag['property_value']
cases = {
    'LTV absent': ltv.isna(), 'LTV > 150': ltv > 150,
    'Population du secteur = 0': diag['tract_population'].eq(0), 'Revenu régional = 0': diag['ffiec_msa_md_median_family_income'].eq(0),
    'Montant du prêt / valeur du bien > 2': loan_value > 2, 'Revenu ≤ 0': diag['income'] <= 0,
}
rows = []
for name, mask in cases.items():
    rows.append({'Cas': name, 'Lignes': int(mask.sum()), 'Acceptation observée': diag.loc[mask, 'accepted'].mean(),
                 'Probabilité moyenne prédite': diag.loc[mask, 'prob'].mean(),
                 'Résidu z': residual[mask].sum() / np.sqrt(variance[mask].sum())})
display_named(pd.DataFrame(rows).round(3))
for threshold in [0.95, 0.98]:
    wrong = ((diag['prob'] > threshold) & diag['accepted'].eq(0)) | ((diag['prob'] < 1 - threshold) & diag['accepted'].eq(1))
    print(f'Erreurs avec confiance ≥ {threshold:.0%} : {int(wrong.sum())} lignes ({wrong.mean():.2%})')

key_columns = ['county_code', 'tract_population', 'tract_minority_population_percent',
               'tract_to_msa_income_percentage', 'tract_owner_occupied_units', 'loan_amount', 'income', 'applicant_age']
# pandas 3 preserves NaN after astype(str); keep the earlier text-key convention.
both = pd.concat([X_base[key_columns], X_test_base[key_columns]], ignore_index=True).astype(str).fillna('nan').agg('|'.join, axis=1)
group_size = both.map(both.value_counts()).iloc[:len(X_base)].to_numpy()
print(f'Lignes train identiques à une autre demande sur secteur + montant + revenu + âge : {(group_size >= 2).mean():.2%}')
'''

BLEND_SEARCH = '''
from scipy.special import logit

reference = read_oof(CV_DIR, MODEL_DEPTH6_INITIAL)
pool = {}
for base in FINAL_MODEL_NAMES:
    pool[f'{base} x3 seeds'] = np.mean([read_oof(CV_DIR, base + t)['prob'] for t in SEED_TAGS.values()], axis=0)
pool[ENSEMBLE_MULTISEED] = np.mean([pool[f'{b} x3 seeds'] for b in FINAL_MODEL_NAMES], axis=0)
pool[MODEL_INITIAL_POOL_LABEL] = reference['prob'].to_numpy()
pool['LightGBM'] = np.mean([read_oof(LGBM_DIR, g)['prob'] for g in [LIGHTGBM_LEAVES31, LIGHTGBM_LEAVES127]], axis=0)
cpu = pd.read_csv(ARTIFACTS / '03_model_selection' / 'baseline_catboost_xgboost' / 'oof_predictions.csv')
cpu = cpu.sort_values('id', kind='stable').reset_index(drop=True)
require(cpu['id'].equals(reference['id']) and cpu['fold'].equals(reference['fold']), 'Prédictions OOF des baselines CPU non alignées')
for column in [BASELINE_CATBOOST_DEPTH6, BASELINE_CATBOOST_DEPTH8, 'XGBoost']:
    pool[f'CPU {column}'] = cpu[f'prob_{column}'].to_numpy()
names = list(pool)
labels = reference['accepted'].to_numpy()
folds = reference['fold'].to_numpy()
logits = np.column_stack([logit(np.clip(pool[n], 1e-6, 1 - 1e-6)) for n in names])
display_named(pd.DataFrame({'AUC sur cinq folds': [fold_auc(reference, pool[n]).mean() for n in names],
                      'Corrélation avec l’ensemble multigraines': [np.corrcoef(pool[n], pool[ENSEMBLE_MULTISEED])[0, 1] for n in names]},
                     index=names).sort_values('AUC sur cinq folds', ascending=False).round(5))


def mean_fold_auc(score, use_folds):
    return np.mean([roc_auc_score(labels[folds == k], score[folds == k]) for k in use_folds])


def greedy(use_folds, steps=15):
    start = names.index(ENSEMBLE_MULTISEED)
    chosen, current = [start], logits[:, start].copy()
    best = mean_fold_auc(current, use_folds)
    for _ in range(steps):
        pick = None
        for j in range(len(names)):
            trial = (current * len(chosen) + logits[:, j]) / (len(chosen) + 1)
            score = mean_fold_auc(trial, use_folds)
            if score > best + 1e-6:
                best, pick = score, j
        if pick is None:
            break
        current = (current * len(chosen) + logits[:, pick]) / (len(chosen) + 1)
        chosen.append(pick)
    return chosen


rows = []
for held_out in range(1, 6):
    chosen = greedy([k for k in range(1, 6) if k != held_out])
    blend = logits[:, chosen].mean(axis=1)
    mask = folds == held_out
    rows.append({'Fold réservé': held_out,
                 'AUC de l’ensemble': roc_auc_score(labels[mask], blend[mask]),
                 'l’ensemble multigraines AUC': roc_auc_score(labels[mask], pool[ENSEMBLE_MULTISEED][mask]),
                 'Modèles sélectionnés': ', '.join(f'{names[c]}×{chosen.count(c)}' for c in sorted(set(chosen)))})
nested = pd.DataFrame(rows)
nested['Écart'] = nested['AUC de l’ensemble'] - nested['l’ensemble multigraines AUC']
display_named(nested.round(5))
print(f"Validation imbriquée : gain moyen {nested['Écart'].mean():+.5f}, {int((nested['Écart'] > 0).sum())}/5 folds améliorés")
'''

PSEUDO_RESULTS = '''
PL_DIR = ARTIFACTS / '05_error_analysis' / 'pseudo_label'
pl_runs = load_runs(PL_DIR, PSEUDO_LABEL_RUN_PATTERN)
pl_table = pl_runs.pivot(index='fold', columns='name', values='valid_auc')
pl_table['gain'] = pl_table[SOFT_PSEUDO_LABELS] - pl_table[OBSERVED_LABELS_ONLY]
display_named(pl_table.round(6))
print(f"Pseudo-labels souples : gain moyen {pl_table['gain'].mean():+.5f}, {int((pl_table['gain'] > 0).sum())}/5 folds positifs")
'''

NOISE = '''
labels = reference['accepted'].to_numpy()
new, old = pool[ENSEMBLE_MULTISEED], pool[MODEL_INITIAL_POOL_LABEL]
rng = np.random.default_rng(SEED)
subset_auc, paired = [], []
for _ in range(300):
    idx = rng.choice(len(labels), 13500, replace=False)
    auc_new = roc_auc_score(labels[idx], new[idx])
    subset_auc.append(auc_new)
    paired.append(auc_new - roc_auc_score(labels[idx], old[idx]))
true_gap = roc_auc_score(labels, new) - roc_auc_score(labels, old)
print(f'Écart l’ensemble multigraines − la référence initiale sur les 105 000 lignes : {true_gap:+.5f}')
print(f'Sur 13 500 lignes aléatoires (≈ classement public) : écart-type AUC {np.std(subset_auc):.4f} ; écart-type de la différence appariée {np.std(paired):.4f} ; '
      f'fréquence d’inversion dans la simulation {np.mean(np.array(paired) < 0):.0%}')
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(np.array(paired) * 1e4, bins=30, color='#4c72b0')
ax.axvline(true_gap * 1e4, color='black', ls='--', label='écart sur toutes les lignes')
ax.axvline(0, color='#c44e52', lw=1)
ax.set_xlabel('AUC(l’ensemble multigraines) − AUC(la référence initiale), 13 500 lignes aléatoires (×1e-4)')
ax.set_title('Variabilité d’un classement public sur 30 % du test')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

lc = load_runs(FULL_DIR, LEARNING_CURVE_FULL_PATTERN).set_index('fold')['valid_auc']
reduced_model_runs = load_runs(CV_DIR, REDUCED_MODEL_RUN_PATTERN).set_index('fold')['valid_auc']
display_named(pd.DataFrame({'Courbe d’apprentissage, 84 000 lignes': lc, 'Modèle réduit en validation croisée': reduced_model_runs, 'Écart': lc - reduced_model_runs}).round(6))
'''

LGBM_RESULTS = '''
LGBM_DIR = ARTIFACTS / '05_error_analysis' / 'lightgbm'
lgbm_runs = load_runs(LGBM_DIR, LIGHTGBM_RUN_PATTERN)
display_named(lgbm_runs.pivot(index='fold', columns='group', values='valid_auc').agg(['mean', 'std']).round(5))

cv_reference = read_oof(CV_DIR, MODEL_DEPTH6_INITIAL)
multiseed_ensemble = np.mean([np.mean([read_oof(CV_DIR, b + t)['prob'] for t in SEED_TAGS.values()], axis=0)
                   for b in FINAL_MODEL_NAMES], axis=0)
lgbm_oof = [read_oof(LGBM_DIR, g) for g in sorted(lgbm_runs['group'].unique())]
for frame in lgbm_oof:
    require(frame['id'].equals(cv_reference['id']) and frame['fold'].equals(cv_reference['fold']),
            'Prédictions OOF LightGBM et CatBoost non alignées')
lgbm = np.mean([frame['prob'] for frame in lgbm_oof], axis=0)
base_folds = fold_auc(cv_reference, multiseed_ensemble)
blend_rows = []
for weight in [0.0, 0.1, 0.2, 0.3, 1.0]:                  # Poids prédéfinis, sans recherche
    folds = fold_auc(cv_reference, (1 - weight) * multiseed_ensemble + weight * lgbm)
    blend_rows.append({'Poids LightGBM': weight, 'AUC sur cinq folds': folds.mean(),
                       'Écart à l’ensemble multigraines': (folds - base_folds).mean(),
                       'Folds améliorés': int((folds - base_folds).gt(0).sum())})
display_named(pd.DataFrame(blend_rows).round(6))
print(f'Corrélation OOF de l’ensemble multigraines et LightGBM : {np.corrcoef(multiseed_ensemble, lgbm)[0, 1]:.4f}')
'''

TABM_RESULTS = '''
NN_DIR = ARTIFACTS / '05_error_analysis' / 'tabm'
tabm_runs = load_runs(NN_DIR, 'TabMMini__fold?.json')
display_named(tabm_runs[['fold', 'valid_auc', 'best_epoch', 'inner_stop_auc']].round(6))
display_named(pd.read_csv(NN_DIR / 'blend_by_fold.csv').round(6))
'''

TABM_SETUP = '''
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED)
np.random.seed(SEED)
'''

nb05 = [
    md(''),
    code(SETUP),
    naming_compatibility_cell(),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'lightgbm': False,       # LightGBM, cinq folds (CPU, environ 30 min)
    'tabm': False,           # TabM-mini (GPU, environ 5 min pour le fold 1)
    'pseudo_label': False,   # Pseudo-labels par fold (GPU, environ 2 h ; étape complète si validation favorable)
}
'''),
    md(''),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS + '\n\n' + OOF_HELPERS),
    md(''),
    code(RESIDUAL),
    md(''),
    md(''),
    code(ANOMALY),
    md(''),
    raw_code(cell('lgbm_config')),
    gated("TRAIN['lightgbm']", cell('lgbm_run'), gpu=False, catboost=False,
          setup='OUTPUT_DIR.mkdir(parents=True, exist_ok=True)'),
    code(LGBM_RESULTS),
    md(''),
    md(''),
    raw_code(cell('tabm_config')),
    gated("TRAIN['tabm']", cell('tabm_prep') + '\n\n' + cell('tabm_run') + '\n\n' + cell('tabm_blend'),
          catboost=False, setup=TABM_SETUP),
    code(TABM_RESULTS),
    md(''),
    md(''),
    code(BLEND_SEARCH),
    md(''),
    md(''),
    raw_code(cell('pseudo_config')),
    gated("TRAIN['pseudo_label']", cell('pseudo_functions') + '\n\n' + cell('pseudo_validate') + '\n\n' + cell('pseudo_final')),
    code(PSEUDO_RESULTS),
    md(''),
    md(''),
    code(NOISE),
    md(''),
    md(''),
]

# ============================================================================= 06
nb06 = [
    md(''),
    code(SETUP),
    naming_compatibility_cell(),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'protected_ablation': False,   # référence et modèle sans variables protégées, cinq folds, valeurs de Shapley (GPU, environ 75 min)
}
'''),
    md(''),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS + '\n\n' + OOF_HELPERS),
    md(''),
    raw_code(cell('interp_config')),
    gated("TRAIN['protected_ablation']", cell('interp_run')),
    md(''),
    raw_code(cell('interp_shap')),
    md(''),
    md(''),
    raw_code(cell('interp_calibration')),
    md(''),
    md(''),
    raw_code(cell('interp_groups')),
    md(''),
    md(''),
    raw_code(cell('interp_ablation')),
    md(''),
    md(''),
]

save(nb01, '01_data_exploration.ipynb')
save(nb02, '02_feature_engineering.ipynb')
save(nb03, '03_model_selection.ipynb')
save(nb04, '04_final_model.ipynb')
save(nb05, '05_error_analysis.ipynb')
save(nb06, '06_interpretation.ipynb')
