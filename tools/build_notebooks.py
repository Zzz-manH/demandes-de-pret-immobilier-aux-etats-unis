"""生成 Colab notebook 09–19（以及已删除的 08、11，保留其生成代码以便追溯）。

用法：python tools/build_notebooks.py <输出目录> [只生成的编号，如 17]
注意：输出到项目根目录会覆盖带运行输出的 notebook，修改时请先输出到其他目录再复制。
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

project = Path(sys.argv[1])
ONLY = set(sys.argv[2:])
SHARED = Path(__file__).resolve().parent / 'shared'
DATA_CELL = (SHARED / 'data_contract_cell.py').read_text(encoding='utf-8')   # 数据契约校验（原 04 的单元格）
DTI_CELL = (SHARED / 'dti_bounds_cell.py').read_text(encoding='utf-8')       # DTI 上下界（原 04 的单元格）

SETUP_COMMON = '''from pathlib import Path
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

try:
    from google.colab import drive
except ModuleNotFoundError:
    drive = None
if drive is not None:
    drive.mount('/content/drive')

PROJECT_DIR = Path(os.environ.get(
    'HMDA_PROJECT_DIR',
    '/content/drive/MyDrive/Colab Notebooks/demandes-de-pret-immobilier-aux-etats-unis',
)).expanduser().resolve()
PREPARED_DIR = PROJECT_DIR / 'artifacts' / 'prepared'
SEED = 42
CATBOOST_VERSION = '1.2.10'
if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(f'找不到项目目录：{PROJECT_DIR}；请修改 PROJECT_DIR')

try:
    installed_catboost = importlib.metadata.version('catboost')
except importlib.metadata.PackageNotFoundError:
    installed_catboost = None
if installed_catboost != CATBOOST_VERSION:
    if 'catboost' in sys.modules:
        raise RuntimeError('当前运行时已经导入另一版 CatBoost；请重启 Colab 运行时后从头运行')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        f'catboost=={CATBOOST_VERSION}',
    ])
    importlib.invalidate_caches()

import catboost
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

if catboost.__version__ != CATBOOST_VERSION:
    raise RuntimeError(f'需要 CatBoost {CATBOOST_VERSION}；实际为 {catboost.__version__}')
'''

HELPERS = '''

def checked_probabilities(model, features):
    require(list(model.classes_) == [0, 1], '类别顺序不是 [0, 1]')
    probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    require(probability.ndim == 1 and len(probability) == len(features)
            and np.isfinite(probability).all()
            and ((probability >= 0) & (probability <= 1)).all(),
            '预测概率无效')
    return probability


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def fixed_config(output_dir, config):
    """首次运行写入配置；之后配置必须完全相同，避免把不同实验的结果混在一个目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'config.json'
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf-8'))
        require(saved == config,
                f'{output_dir.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
    else:
        write_json(path, config)
'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(cells, name):
    if ONLY and name[:2] not in ONLY:
        return
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'accelerator': 'None' if name.startswith('08') else 'GPU',
        'colab': {'provenance': []},
    }
    nbf.validate(nb)
    nbf.write(nb, project / name)
    print('wrote', project / name)


# --------------------------------------------------------------------- 08
cells08 = [
    md('''
# 08 · HMDA：CPU CatBoost R1/R2 全量训练 + 多种子融合

目的：把**五折验证最好的方案**变成提交文件。五折中 CPU 版 CatBoost 明显好于 GPU 版：

|方案|五折平均 AUC|
|---|---:|
|CPU R1 50% + R2 50%|**0.91608**|
|CPU R2|0.91604|
|GPU R2（当前公开榜 0.91857 的提交）|0.91498|

本 notebook 不重新做五折，只用与正式五折**完全相同的参数**在全部 105,000 行上训练，并对每个模型做 3 个随机种子平均，最后 R1/R2 等权融合。多种子平均只降低随机性，不改变模型。

**操作**
1. Colab「更改运行时类型」选 **CPU**（不需要 GPU；有 Pro 可选高内存/更多核）。
2. 从上到下运行。每个（模型, 种子）训练完立刻把测试概率存到 Drive 的 `artifacts/final_cpu/<EXPERIMENT_ID>/`。
3. **断线后重新从头运行即可续跑**，已完成的会跳过。
4. 每完成一个模型都会刷新融合提交文件；第一轮（种子 42 的 R2 和 R1）完成后就已经可以提交。

耗时参考：旧 Colab CPU 上 R2 每折约 18 分钟、R1 约 12 分钟（84,000 行）；全量 105,000 行会再长一些。全部 6 次约 2–3 小时，取决于分到的 CPU 核数。
'''),
    md('## 1. 环境与配置'),
    code(SETUP_COMMON + '''
EXPERIMENT_ID = 'r1r2_seeds_v1'
SEEDS = [42, 2027, 31415]         # 第一个种子与正式五折相同
THREADS = os.cpu_count() or 2
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'final_cpu' / EXPERIMENT_ID

COMMON_PARAMS = dict(
    iterations=1600, learning_rate=0.05, l2_leaf_reg=3.0,
    loss_function='Logloss', eval_metric='AUC', nan_mode='Min',
    task_type='CPU', thread_count=THREADS,
    use_best_model=False, allow_writing_files=False, verbose=False,
)
MODEL_PARAMS = {                  # 与 artifacts/models/20260930T194248_452800Z_full 相同
    'CatBoost_R2': dict(COMMON_PARAMS, depth=8),
    'CatBoost_R1': dict(COMMON_PARAMS, depth=6),
}
BLEND_WEIGHTS = {'CatBoost_R1': 0.5, 'CatBoost_R2': 0.5}
print(f'项目：{PROJECT_DIR}')
print(f'CPU 线程：{THREADS}；CatBoost {catboost.__version__}；输出：{OUTPUT_DIR}')
'''),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('''
## 3. 全量训练（可续跑）

顺序为「种子 42 的 R2 → R1 → 种子 2027 的 R2 → R1 → …」，这样最早得到可提交的版本。每次训练后：
- 测试概率写成 `test_<模型>_seed<种子>.csv`，训练信息写成同名 `.json`；
- 重新计算融合：每个模型先对已完成的种子取平均，再按 R1/R2 各 50% 融合，写出 `submission_blend_R1xN_R2xM.csv`。
'''),
    code('''
fixed_config(OUTPUT_DIR, {
    'purpose': 'full-data CPU CatBoost R1/R2, seed-averaged, 50/50 blend',
    'seeds': SEEDS,
    'model_params': {name: {k: v for k, v in p.items() if k != 'thread_count'}
                     for name, p in MODEL_PARAMS.items()},
    'blend_weights': BLEND_WEIGHTS,
    'features': 'prepared 56 raw columns',
    'data_hashes': DATA_HASHES,
})


def done_path(name, seed):
    return OUTPUT_DIR / f'test_{name}_seed{seed}.csv'


def write_blend():
    per_model = {}
    for name in MODEL_PARAMS:
        frames = [pd.read_csv(done_path(name, s), dtype={'id': str})
                  for s in SEEDS if done_path(name, s).is_file()]
        for frame in frames:
            require(frame['id'].equals(sample_submission['id']), f'{name} 测试 ID 顺序不一致')
        if frames:
            per_model[name] = (np.mean([f['accepted'].to_numpy() for f in frames], axis=0),
                               len(frames))
    if set(per_model) != set(BLEND_WEIGHTS):
        return None
    blend = sum(BLEND_WEIGHTS[n] * per_model[n][0] for n in BLEND_WEIGHTS)
    tag = f"R1x{per_model['CatBoost_R1'][1]}_R2x{per_model['CatBoost_R2'][1]}"
    path = OUTPUT_DIR / f'submission_blend_{tag}.csv'
    submission = pd.DataFrame({'id': sample_submission['id'], 'accepted': blend})
    require(np.isfinite(blend).all() and len(submission) == 45000, '融合概率无效')
    submission.to_csv(path, index=False, float_format='%.12f')
    print(f'  → 融合提交已更新：{path.name}', flush=True)
    return path


for seed in SEEDS:
    for name, base_params in MODEL_PARAMS.items():
        path = done_path(name, seed)
        if path.is_file():
            print(f'{name} seed={seed}: 已完成，跳过')
            continue
        started = time.perf_counter()
        model = catboost.CatBoostClassifier(**dict(base_params, random_seed=seed))
        model.fit(X_base, y, cat_features=CATEGORICAL_COLUMNS)
        require(int(model.tree_count_) == 1600, '模型未完成固定的 1600 轮')
        probability = checked_probabilities(model, X_test_base)
        seconds = time.perf_counter() - started
        tmp = path.with_suffix('.tmp')
        pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': probability}).to_csv(
            tmp, index=False, float_format='%.12f')
        tmp.replace(path)
        write_json(path.with_suffix('.json'), {
            'model': name, 'seed': seed, 'train_rows': len(X_base),
            'rounds': int(model.tree_count_), 'fit_seconds': seconds, 'threads': THREADS,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'versions': {'python': platform.python_version(),
                         'catboost': catboost.__version__,
                         'platform': platform.platform()},
        })
        print(f'{name} seed={seed}: 完成，{seconds/60:.1f} 分钟', flush=True)
        del model, probability
        gc.collect()
        write_blend()

final_path = write_blend()
print('全部完成。最终提交文件：', final_path)
'''),
    md('''
## 4. 检查种子之间的差异

同一模型不同种子的测试预测相关性很高是正常的；平均后主要减少随机波动。这里只做描述，不用于选模型。
'''),
    code('''
preds = {}
for name in MODEL_PARAMS:
    for seed in SEEDS:
        if done_path(name, seed).is_file():
            preds[f'{name}_s{seed}'] = pd.read_csv(done_path(name, seed))['accepted'].to_numpy()
if preds:
    display(pd.DataFrame(preds).corr(method='spearman').round(5))
display(pd.DataFrame(sorted(p.name for p in OUTPUT_DIR.glob('submission_*.csv')),
                     columns=['可提交文件']))
'''),
]
save(cells08, '08_hmda_colab_cpu_final_blend.ipynb')

# --------------------------------------------------------------------- 09
cells09 = [
    md('''
# 09 · HMDA：Colab GPU 上筛选 CatBoost 参数（第 1 折）

现有 R1/R2 都是**固定学习率 0.05、1600 轮**，从未调过。这里在 GPU 上快速比较更小学习率 + 早停、不同深度、L2 正则、行抽样和 DTI 上下界。

**协议（外层验证不参与任何选择）**
- 外层：固定 seed=42 五折中的第 1 折；84,000 行训练、21,000 行验证。
- 早停候选：在外层训练部分再分层留出 15% 做早停，得到最佳轮数 *b*；然后在整个外层训练部分用 `round(b / 0.85)` 轮重新训练，再在外层验证上评分。
- `B0` 是现有 GPU R2（固定 1600 轮）的同折对照；GPU 上这一折以前是 0.90915。

**判读**：单折只用于筛选。挑出相对 B0 提升最大的 1–2 个，把 `FOLDS` 改成 `[1, 2, 3, 4, 5]`、`RUN_ONLY` 填这几个名字，再跑完整五折；五折平均提升 ≥ 0.0005 且多数折同向才采用。GPU 与 CPU 的绝对分数不同（CPU 约高 0.001），所以这里看**相对 B0 的差**，选出的参数最后用 CPU 全量训练提交（08 同样的做法）。

**操作**：Colab 选 **T4 GPU**，从上到下运行。每个（候选, 折）完成后立刻存到 Drive `artifacts/gpu_tuning/<EXPERIMENT_ID>/`；断线后从头运行可续跑。预计全部候选第 1 折共约 1.5–2.5 小时，便宜的先跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(SETUP_COMMON + '''
if shutil.which('nvidia-smi') is None:
    raise RuntimeError('未找到 nvidia-smi：请在 Colab 更改运行时类型，选择 T4 GPU 后从头运行')
gpu_probe = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    text=True, capture_output=True, check=False,
)
if gpu_probe.returncode != 0 or not gpu_probe.stdout.strip():
    raise RuntimeError('NVIDIA GPU 不可用：请在 Colab 选择 T4 GPU 后从头运行')
GPU_NAME = gpu_probe.stdout.strip().splitlines()[0]

EXPERIMENT_ID = 'catboost_screen_v1'
FOLDS = [1, 2, 3, 4, 5]     # 第二轮：五折复核；只做第 1 折筛选时改回 [1]
RUN_ONLY = ['B0_R2_fixed1600', 'C7_d8_lr03_l2_10_sub80',
            'C10_d8_lr03_l2_30_sub80', 'C11_d6_lr03_l2_10_sub80']  # None 表示全部候选
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'gpu_tuning' / EXPERIMENT_ID

GPU_BASE = dict(
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    allow_writing_files=False, verbose=False,
)
# early_stop=False 表示固定轮数、不早停；按预计耗时从小到大排列
CANDIDATES = {
    'B0_R2_fixed1600':   dict(depth=8, learning_rate=0.05, l2_leaf_reg=3, early_stop=False, iterations=1600),
    'C1_d8_lr05_l2_3_es': dict(depth=8, learning_rate=0.05, l2_leaf_reg=3),
    'C6_d6_lr03_l2_10':  dict(depth=6, learning_rate=0.03, l2_leaf_reg=10),
    'C3_d8_lr03_l2_10':  dict(depth=8, learning_rate=0.03, l2_leaf_reg=10),
    'C2_d8_lr03_l2_3':   dict(depth=8, learning_rate=0.03, l2_leaf_reg=3),
    'C4_d8_lr03_l2_30':  dict(depth=8, learning_rate=0.03, l2_leaf_reg=30),
    'C7_d8_lr03_l2_10_sub80': dict(depth=8, learning_rate=0.03, l2_leaf_reg=10,
                                   bootstrap_type='Bernoulli', subsample=0.8),
    'C8_d8_lr03_l2_10_DTI': dict(depth=8, learning_rate=0.03, l2_leaf_reg=10, dti=True),
    'C5_d10_lr03_l2_10': dict(depth=10, learning_rate=0.03, l2_leaf_reg=10),
    'C9_d8_lr02_l2_10':  dict(depth=8, learning_rate=0.02, l2_leaf_reg=10),
    # 第二轮：把第 1 折前三名（抽样、depth 6、L2=30）组合起来
    'C10_d8_lr03_l2_30_sub80': dict(depth=8, learning_rate=0.03, l2_leaf_reg=30,
                                    bootstrap_type='Bernoulli', subsample=0.8),
    'C11_d6_lr03_l2_10_sub80': dict(depth=6, learning_rate=0.03, l2_leaf_reg=10,
                                    bootstrap_type='Bernoulli', subsample=0.8),
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(CANDIDATES)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知候选：{sorted(unknown)}')
print(f'项目：{PROJECT_DIR}')
print(f'GPU：{GPU_NAME}；CatBoost {catboost.__version__}；折：{FOLDS}；输出：{OUTPUT_DIR}')
'''),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同，只给 `dti=True` 的候选使用）'),
    code(DTI_CELL),
    md('''
## 4. 训练与评分（可续跑）

每个（候选, 折）结果写为 `<候选>__fold<k>.json`，验证概率写为同名 `.csv`。配置里只记录候选定义，所以**以后新增候选不会影响已完成结果**；但修改已有候选的参数时必须改名或改 `EXPERIMENT_ID`。
'''),
    code('''
fixed_config(OUTPUT_DIR, {
    'purpose': 'GPU CatBoost screening, inner 15% early stop then outer refit',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def candidate_params(spec, iterations):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'dti', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations)


def run_one(name, spec, fold):
    features = X_dti if spec.get('dti') else X_base
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    best_iteration = None
    early_stopped = None
    if spec.get('early_stop', True):
        inner_fit, inner_stop = train_test_split(
            fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED,
            stratify=y.iloc[fit_idx],
        )
        probe = catboost.CatBoostClassifier(**candidate_params(spec, MAX_ITERATIONS),
                                            eval_metric='Logloss', od_type='Iter',
                                            od_wait=OD_WAIT, use_best_model=True)
        probe.fit(features.iloc[inner_fit], y.iloc[inner_fit],
                  cat_features=CATEGORICAL_COLUMNS,
                  eval_set=(features.iloc[inner_stop], y.iloc[inner_stop]))
        best_iteration = int(probe.get_best_iteration()) + 1
        early_stopped = int(probe.tree_count_) < MAX_ITERATIONS
        if not early_stopped:
            print(f'  注意：{name} 到 {MAX_ITERATIONS} 轮仍未早停，结果可能偏保守')
        rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
        del probe
        gc.collect()
    else:
        rounds = int(spec['iterations'])
    model = catboost.CatBoostClassifier(**candidate_params(spec, rounds), use_best_model=False)
    model.fit(features.iloc[fit_idx], y.iloc[fit_idx], cat_features=CATEGORICAL_COLUMNS)
    require(int(model.tree_count_) == rounds, '外层重训轮数不符')
    probability = checked_probabilities(model, features.iloc[valid_idx])
    actual = y.iloc[valid_idx]
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': probability}).to_csv(
        OUTPUT_DIR / f'{name}__fold{fold}.csv', index=False, float_format='%.12f')
    result = {
        'candidate': name, 'fold': fold,
        'spec': {k: v for k, v in spec.items()},
        'feature_count': features.shape[1],
        'inner_best_iteration': best_iteration, 'early_stopped': early_stopped, 'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, probability),
        'valid_logloss': log_loss(actual, probability, labels=[0, 1]),
        'valid_brier': brier_score_loss(actual, probability),
        'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUTPUT_DIR / f'{name}__fold{fold}.json', result)
    del model
    gc.collect()
    return result


def load_results():
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))]
    return pd.DataFrame(rows)


selected = {n: s for n, s in CANDIDATES.items() if RUN_ONLY is None or n in RUN_ONLY}
for fold in FOLDS:
    for name, spec in selected.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['spec'] == spec, f'{name} 的定义已改变；请改候选名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        result = run_one(name, spec, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'轮数={result["refit_rounds"]}  {result["seconds"]/60:.1f} 分钟', flush=True)
'''),
    md('''
## 5. 与 B0 配对比较

`delta_vs_B0` 是同一折上的 AUC 差。只有同时跑了相同折的候选才会出现在五折汇总里。
'''),
    code('''
results = load_results()
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='candidate', values='valid_auc')
require('B0_R2_fixed1600' in auc.columns, '缺少对照 B0_R2_fixed1600')
delta = auc.sub(auc['B0_R2_fixed1600'], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    'mean_delta_vs_B0': delta.mean(),
    'folds_better_than_B0': (delta > 0).sum(),
    'mean_refit_rounds': results.groupby('candidate')['refit_rounds'].mean(),
    'mean_logloss': results.groupby('candidate')['valid_logloss'].mean(),
    'total_minutes': results.groupby('candidate')['seconds'].sum() / 60,
}).sort_values('mean_delta_vs_B0', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
'''),
]
save(cells09, '09_hmda_colab_gpu_catboost_tuning.ipynb')

# --------------------------------------------------------------------- 10
GPU_CHECK = '''
if shutil.which('nvidia-smi') is None:
    raise RuntimeError('未找到 nvidia-smi：请在 Colab 更改运行时类型，选择 T4 GPU 后从头运行')
gpu_probe = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    text=True, capture_output=True, check=False,
)
if gpu_probe.returncode != 0 or not gpu_probe.stdout.strip():
    raise RuntimeError('NVIDIA GPU 不可用：请在 Colab 选择 T4 GPU 后从头运行')
GPU_NAME = gpu_probe.stdout.strip().splitlines()[0]
'''

cells10 = [
    md('''
# 10 · HMDA：特征块消融（Colab GPU，五折）

数据检查（2026-10-14）没有发现数据错误：训练/测试对抗验证 AUC 0.501、无冲突重复、ID 无泄漏。因此这里不「清洗」，而是检验**给模型加入它不容易自己算出的信息**能否提高五折 AUC。

|特征块|新增列|想法|
|---|---|---|
|`dti`|`dti_lower`、`dti_upper`|DTI 类别的数值上下界（与 04 相同）|
|`ltv`|`ltv_calc`、`ltv_gap`|由贷款额/房价算出的 LTV，以及与申报 LTV 的差|
|`income`|`income_to_area`、`loan_to_income`、`value_to_income`|收入相对地区收入、贷款额和房价相对收入|
|`missing`|`missing_pattern`（类别）、`missing_count`|关键字段的缺失组合；LTV 缺失时批准率 54%，有值时 89%|
|`freq`|`freq_county_code`、`freq_derived_msa-md`、`freq_state_code`|地区在训练+测试中出现的次数（不使用标签）|

**协议**：固定 seed=42 五折；每个变体 = 原始 56 列 + 一个特征块，`F6_all` 为全部块。模型统一用快速配置（CatBoost GPU，depth 8、lr 0.05、L2=3、Bernoulli 抽样 0.8、固定 1600 轮，每折约 2 分钟），只比较**同折相对 `F0_base` 的差**。比值的分母必须 > 0，否则为缺失，不加任意 epsilon。

**判读**：五折平均提升 ≥ 0.0003 且至少 4/5 折同向的块才保留；保留的块再合并，换成 09 选出的参数做最终训练。

**操作**：Colab 选 **T4 GPU**，从上到下运行。约 7 个变体 × 5 折 × 2 分钟 ≈ 70–80 分钟；每个（变体, 折）完成即存到 Drive `artifacts/feature_ablation/<EXPERIMENT_ID>/`，断线后从头运行可续跑。可以和 09 同时在另一个 Colab 窗口运行。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(SETUP_COMMON + GPU_CHECK + '''
EXPERIMENT_ID = 'feature_blocks_v1'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None             # 例如 ['F0_base', 'F2_ltv']；None 表示全部变体
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'feature_ablation' / EXPERIMENT_ID

FAST_PARAMS = dict(
    iterations=1600, depth=8, learning_rate=0.05, l2_leaf_reg=3,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    use_best_model=False, allow_writing_files=False, verbose=False,
)
VARIANTS = {
    'F0_base': [],
    'F1_dti': ['dti'],
    'F2_ltv': ['ltv'],
    'F3_income': ['income'],
    'F4_missing': ['missing'],
    'F5_freq': ['freq'],
    'F6_all': ['dti', 'ltv', 'income', 'missing', 'freq'],
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知变体：{sorted(unknown)}')
print(f'项目：{PROJECT_DIR}')
print(f'GPU：{GPU_NAME}；CatBoost {catboost.__version__}；折：{FOLDS}；输出：{OUTPUT_DIR}')
'''),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('''
## 4. 构建特征块

训练与测试一起构建，保证口径一致；只有 `freq` 用到测试集的分布（只计数，不涉及标签）。原 56 列保持不变，新列追加在后面。
'''),
    code('''
def safe_ratio(numerator, denominator):
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    return (numerator / denominator.where(valid)).where(valid).astype('float64')


def build_blocks(train, test):
    both = pd.concat([train, test], ignore_index=True)
    blocks = {}
    blocks['dti'] = add_dti_bounds(both)[['dti_lower', 'dti_upper']]

    ltv_calc = 100 * safe_ratio(both['loan_amount'], both['property_value'])
    blocks['ltv'] = pd.DataFrame({
        'ltv_calc': ltv_calc,
        'ltv_gap': both['loan_to_value_ratio'] - ltv_calc,
    })

    income_dollars = both['income'] * 1000      # HMDA 收入单位为千美元
    blocks['income'] = pd.DataFrame({
        'income_to_area': safe_ratio(income_dollars, both['ffiec_msa_md_median_family_income']),
        'loan_to_income': safe_ratio(both['loan_amount'], income_dollars),
        'value_to_income': safe_ratio(both['property_value'], income_dollars),
    })

    missing = pd.DataFrame({
        'ltv': both['loan_to_value_ratio'].isna(),
        'pv': both['property_value'].isna(),
        'inc': both['income'].isna(),
        'term': both['loan_term'].isna(),
        'dti': both['debt_to_income_ratio'].eq('MISSING'),
        'age': both['applicant_age'].eq('MISSING'),
    })
    blocks['missing'] = pd.DataFrame({
        'missing_pattern': 'm' + missing.astype(int).astype(str).agg(''.join, axis=1),
        'missing_count': missing.sum(axis=1).astype('float64'),
    })

    freq = {}
    for column in ['county_code', 'derived_msa-md', 'state_code']:
        counts = both[column].map(both[column].value_counts()).astype('float64')
        freq[f'freq_{column}'] = counts.where(both[column].ne('MISSING'))
    blocks['freq'] = pd.DataFrame(freq)

    n_train = len(train)
    out = {}
    for name, frame in blocks.items():
        require(len(frame) == len(both), f'{name} 行数不符')
        out[name] = (frame.iloc[:n_train].reset_index(drop=True),
                     frame.iloc[n_train:].reset_index(drop=True))
    return out


BLOCKS = build_blocks(X_base, X_test_base)
NEW_CATEGORICAL = {'missing_pattern'}


def variant_frames(blocks):
    train_parts = [X_base] + [BLOCKS[b][0] for b in blocks]
    test_parts = [X_test_base] + [BLOCKS[b][1] for b in blocks]
    train = pd.concat(train_parts, axis=1)
    test = pd.concat(test_parts, axis=1)
    require(train.columns.is_unique and train.columns.equals(test.columns), '变体列不一致')
    require(train.iloc[:, :len(FEATURE_COLUMNS)].equals(X_base), '原始 56 列被修改')
    cats = CATEGORICAL_COLUMNS + [c for c in train.columns if c in NEW_CATEGORICAL]
    return train, test, cats


for name, (train_block, _) in BLOCKS.items():
    described = train_block.describe(include='all').T
    print(f'\\n[{name}] 缺失率：', train_block.isna().mean().round(3).to_dict())
    display(described.iloc[:, :6])
'''),
    md('''
## 5. 五折训练（可续跑）

每个（变体, 折）结果写为 `<变体>__fold<k>.json` 和验证概率 `.csv`。修改已有变体的定义时必须改名或改 `EXPERIMENT_ID`。
'''),
    code('''
fixed_config(OUTPUT_DIR, {
    'purpose': 'feature block ablation, fast GPU CatBoost, paired vs F0_base',
    'seed': SEED,
    'fast_params': {k: v for k, v in FAST_PARAMS.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})

selected = {n: b for n, b in VARIANTS.items() if RUN_ONLY is None or n in RUN_ONLY}
for fold in FOLDS:
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    for name, blocks in selected.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['blocks'] == blocks, f'{name} 的定义已改变；请改变体名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        train, _, cats = variant_frames(blocks)
        started = time.perf_counter()
        model = catboost.CatBoostClassifier(**FAST_PARAMS)
        model.fit(train.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats)
        require(int(model.tree_count_) == FAST_PARAMS['iterations'], '轮数不符')
        probability = checked_probabilities(model, train.iloc[valid_idx])
        actual = y.iloc[valid_idx]
        pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                      'prob': probability}).to_csv(
            OUTPUT_DIR / f'{name}__fold{fold}.csv', index=False, float_format='%.12f')
        result = {
            'variant': name, 'blocks': blocks, 'fold': fold,
            'feature_count': train.shape[1],
            'valid_auc': roc_auc_score(actual, probability),
            'valid_logloss': log_loss(actual, probability, labels=[0, 1]),
            'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        write_json(result_path, result)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'{train.shape[1]} 列  {result["seconds"]/60:.1f} 分钟', flush=True)
        del model, train
        gc.collect()
'''),
    md('''
## 6. 与 F0_base 配对比较

`keep` 为建议：五折都完成、平均提升 ≥ 0.0003 且至少 4/5 折优于 F0。
'''),
    code('''
results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))])
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='variant', values='valid_auc')
require('F0_base' in auc.columns, '缺少基线 F0_base')
delta = auc.sub(auc['F0_base'], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    'mean_delta_vs_F0': delta.mean(),
    'folds_better_than_F0': (delta > 0).sum(),
    'mean_logloss': results.groupby('variant')['valid_logloss'].mean(),
    'feature_count': results.groupby('variant')['feature_count'].first(),
})
table['keep'] = ((table['folds_done'] == 5) & (table['mean_delta_vs_F0'] >= 0.0003)
                 & (table['folds_better_than_F0'] >= 4))
table.loc['F0_base', 'keep'] = False
table = table.sort_values('mean_delta_vs_F0', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
'''),
]
save(cells10, '10_hmda_colab_gpu_feature_ablation.ipynb')

# --------------------------------------------------------------------- 11
BLOCKS_CELL = cells10[8]['source']

cells11 = [
    md('''
# 11 · HMDA：最佳参数 × 最佳特征，五折确认并导出提交（Colab GPU）

前两步的五折证据：

|来源|结果|
|---|---|
|10 特征消融|`F6_all`（全部特征块）相对原始 56 列 **+0.0030，5/5 折**；`F2_ltv` +0.0021，5/5；`F5_freq` 单独 −0.0001|
|09 参数（前 4 折）|C7（depth 8、lr 0.03、L2=10、Bernoulli 0.8、早停）相对 B0 **+0.0017，4/4**；C11（depth 6）+0.0015|

本 notebook 在同一运行中比较：

|组|参数|特征|
|---|---|---|
|`K0_fast_all`|10 的快速配置（固定 1600 轮）|F6_all（对照）|
|`K1_C7_all`|C7|F6_all|
|`K2_C11_all`|C11|F6_all|
|`K3_C7_nofreq`|C7|F6_all 去掉地区频次|

**协议**与 09 相同：外层固定五折；早停只用外层训练部分内再留出的 15%，然后在整个外层训练部分按 `round(b / 0.85)` 轮重训，外层验证只用于评分。

**提交**：每个外层折模型同时预测测试集；一组五折都完成后，五个折模型的测试概率取平均，写出 `submission_<组>.csv`。这样不需要再做全量训练，也不用外推轮数。**选择五折平均 AUC 最高的那组提交**，不看公开榜。

**操作**：Colab 选 **T4 GPU**，从上到下运行。早停组每折约 7–9 分钟，全部约 2–2.5 小时；每个（组, 折）完成即存 Drive `artifacts/final_gpu/<EXPERIMENT_ID>/`，断线后从头运行可续跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(SETUP_COMMON + GPU_CHECK + '''
EXPERIMENT_ID = 'params_x_features_v1'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None
INNER_STOP_FRACTION = 0.15
MAX_ITERATIONS = 10000
OD_WAIT = 300
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'final_gpu' / EXPERIMENT_ID

GPU_BASE = dict(
    loss_function='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    allow_writing_files=False, verbose=False,
)
ALL_BLOCKS = ['dti', 'ltv', 'income', 'missing', 'freq']
NO_FREQ = ['dti', 'ltv', 'income', 'missing']
FAST = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3,
            bootstrap_type='Bernoulli', subsample=0.8, early_stop=False, iterations=1600)
C7 = dict(depth=8, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
C11 = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, bootstrap_type='Bernoulli', subsample=0.8)
GROUPS = {
    'K0_fast_all': dict(params=FAST, blocks=ALL_BLOCKS),
    'K1_C7_all': dict(params=C7, blocks=ALL_BLOCKS),
    'K2_C11_all': dict(params=C11, blocks=ALL_BLOCKS),
    'K3_C7_nofreq': dict(params=C7, blocks=NO_FREQ),
}
BASELINE = 'K0_fast_all'
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(GROUPS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知组：{sorted(unknown)}')
print(f'项目：{PROJECT_DIR}')
print(f'GPU：{GPU_NAME}；CatBoost {catboost.__version__}；折：{FOLDS}；输出：{OUTPUT_DIR}')
'''),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 构建特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('''
## 5. 五折训练 + 测试预测（可续跑）

每个（组, 折）写出 `<组>__fold<k>.json`、验证概率 `__valid.csv` 和测试概率 `__test.csv`。修改已有组的定义时必须改名或改 `EXPERIMENT_ID`。
'''),
    code('''
fixed_config(OUTPUT_DIR, {
    'purpose': 'params x feature blocks, 5-fold with inner early stop; fold-averaged test predictions',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def model_params(spec, iterations):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations)


def run_one(name, group, fold):
    spec = group['params']
    train, test, cats = variant_frames(group['blocks'])
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    best_iteration, early_stopped = None, None
    if spec.get('early_stop', True):
        inner_fit, inner_stop = train_test_split(
            fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED,
            stratify=y.iloc[fit_idx],
        )
        probe = catboost.CatBoostClassifier(**model_params(spec, MAX_ITERATIONS),
                                            eval_metric='Logloss', od_type='Iter',
                                            od_wait=OD_WAIT, use_best_model=True)
        probe.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
                  eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
        best_iteration = int(probe.get_best_iteration()) + 1
        early_stopped = int(probe.tree_count_) < MAX_ITERATIONS
        if not early_stopped:
            print(f'  注意：{name} 到 {MAX_ITERATIONS} 轮仍未早停')
        rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
        del probe
        gc.collect()
    else:
        rounds = int(spec['iterations'])
    model = catboost.CatBoostClassifier(**model_params(spec, rounds), use_best_model=False)
    model.fit(train.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats)
    require(int(model.tree_count_) == rounds, '外层重训轮数不符')
    valid_prob = checked_probabilities(model, train.iloc[valid_idx])
    test_prob = checked_probabilities(model, test)
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'{name}__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': test_prob}).to_csv(
        f'{stem}__test.csv', index=False, float_format='%.12f')
    result = {
        'group': name, 'fold': fold, 'definition': group,
        'feature_count': train.shape[1],
        'inner_best_iteration': best_iteration, 'early_stopped': early_stopped,
        'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)   # 最后写 json，作为该折完成的标志
    del model, train, test
    gc.collect()
    return result


selected = {n: g for n, g in GROUPS.items() if RUN_ONLY is None or n in RUN_ONLY}
for fold in FOLDS:
    for name, group in selected.items():
        result_path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            require(saved['definition'] == group, f'{name} 的定义已改变；请改组名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        result = run_one(name, group, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  '
              f'轮数={result["refit_rounds"]}  {result["seconds"]/60:.1f} 分钟', flush=True)
'''),
    md('''
## 6. 五折比较，并为完成五折的组写出提交文件

提交文件 = 五个外层折模型测试概率的平均。表中 `oof_auc` 为拼接五折验证预测的整体 AUC。
'''),
    code('''
results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='group', values='valid_auc')
require(BASELINE in auc.columns, f'缺少对照 {BASELINE}')
delta = auc.sub(auc[BASELINE], axis=0)

rows = []
for name in auc.columns:
    folds_done = sorted(results.loc[results['group'].eq(name), 'fold'])
    row = {'group': name, 'folds_done': len(folds_done),
           'mean_auc': auc[name].mean(), 'mean_delta_vs_K0': delta[name].mean(),
           'folds_better_than_K0': int((delta[name] > 0).sum()),
           'mean_refit_rounds': results.loc[results['group'].eq(name), 'refit_rounds'].mean()}
    if folds_done == [1, 2, 3, 4, 5]:
        valid = pd.concat([pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__valid.csv') for k in folds_done])
        row['oof_auc'] = roc_auc_score(valid['accepted'], valid['prob'])
        tests = [pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__test.csv', dtype={'id': str})
                 for k in folds_done]
        for frame in tests:
            require(frame['id'].equals(sample_submission['id']), f'{name} 测试 ID 顺序不一致')
        submission = pd.DataFrame({
            'id': sample_submission['id'],
            'accepted': np.mean([t['accepted'].to_numpy() for t in tests], axis=0),
        })
        require(np.isfinite(submission['accepted']).all() and len(submission) == 45000,
                '提交概率无效')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        submission.to_csv(path, index=False, float_format='%.12f')
        row['submission'] = path.name
    rows.append(row)
table = pd.DataFrame(rows).set_index('group').sort_values('mean_auc', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
complete = table[table['folds_done'] == 5]
if not complete.empty:
    best = complete['mean_auc'].idxmax()
    print(f'五折平均 AUC 最高：{best}（{complete.loc[best, "mean_auc"]:.6f}）→ 提交 {complete.loc[best, "submission"]}')
'''),
]
save(cells11, '11_hmda_colab_gpu_final.ipynb')

# --------------------------------------------------------------------- 12
SETUP10 = cells10[2]['source']
RUN10 = cells10[10]['source']
SUMMARY10 = cells10[12]['source']

cells12 = [
    md('''
# 12 · HMDA：第二轮特征（Colab GPU，五折）

第一轮（10）最有效的是 `F2_ltv`：由贷款额/房价**重新算出** LTV 并与申报值比较，五折 +0.0021。这一轮沿同一思路，加入模型从原始列**无法通过阈值切分自己得到**的信息。基线改为第一轮最好的 `F6_all`（56 + 12 列）。

|特征块|新增列|依据（训练集观察）|
|---|---|---|
|`precision`|`ltv_precision`（类别：缺失/整数/1–3 位小数/更多）|LTV 申报为整数时批准率 90.7%，两位小数 84.4%；反映不同机构的申报习惯，树无法从数值本身切出「是否整数」|
|`thresholds`|`ltv_band`（类别）、`ltv_over_program`|80、95、97、100 是常见产品门槛；`ltv_over_program` = LTV − 该贷款类型通常上限（常规 97、FHA 96.5、VA/RHS 100）|
|`implied`|`implied_property_value`、`implied_value_ratio`|由贷款额和申报 LTV 反推房价，再与申报房价比较（比值形式，与 `ltv_gap` 互补）|
|`term`|`term_is_360`、`term_years`|贷款期限非 30 年的申请更少见|

另有 `G6_nofreq`（第一轮去掉地区频次）与 `G7_nofreq_round2_all`：11 中去掉频次的 K3 五折与公开榜都略好，这两组检验第二轮特征该加在哪个基础上。

**协议**与 10 相同：快速配置（CatBoost GPU、depth 8、lr 0.05、L2=3、Bernoulli 0.8、1600 轮），固定五折，与 `G0_all` 同折配对。五折平均 ≥ +0.0003 且 ≥ 4/5 折才保留，保留的块之后用 C11 参数重训提交（同 11）。

**操作**：Colab 选 **T4 GPU**，从上到下运行。8 个变体 × 5 折 × 约 2 分钟 ≈ 80–90 分钟；结果存 Drive `artifacts/feature_ablation/<EXPERIMENT_ID>/`，断线后从头运行可续跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(SETUP10.replace("EXPERIMENT_ID = 'feature_blocks_v1'", "EXPERIMENT_ID = 'feature_blocks_round2_v1'").replace(
        """VARIANTS = {
    'F0_base': [],
    'F1_dti': ['dti'],
    'F2_ltv': ['ltv'],
    'F3_income': ['income'],
    'F4_missing': ['missing'],
    'F5_freq': ['freq'],
    'F6_all': ['dti', 'ltv', 'income', 'missing', 'freq'],
}""",
        """ROUND1 = ['dti', 'ltv', 'income', 'missing', 'freq']     # 10 中的 F6_all
NO_FREQ = ['dti', 'ltv', 'income', 'missing']
VARIANTS = {
    'G0_all': ROUND1,
    'G1_precision': ROUND1 + ['precision'],
    'G2_thresholds': ROUND1 + ['thresholds'],
    'G3_implied': ROUND1 + ['implied'],
    'G4_term': ROUND1 + ['term'],
    'G5_round2_all': ROUND1 + ['precision', 'thresholds', 'implied', 'term'],
    # 11 中去掉地区频次（K3）五折与公开榜都略好：检验第二轮特征加在无频次基础上
    'G6_nofreq': NO_FREQ,
    'G7_nofreq_round2_all': NO_FREQ + ['precision', 'thresholds', 'implied', 'term'],
}""")),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('''
## 5. 第二轮特征块

`ltv_precision` 需要申报 LTV 的**原始文本**（例如 `80.0` 与 `79.85`），所以从 01 的准备文件按字符串重新读取这一列；01 已保证它与原始 CSV 逐字相同。
'''),
    code('''
def read_raw_ltv(path):
    text = pd.read_csv(path, usecols=['loan_to_value_ratio'], dtype=str,
                       keep_default_na=False)['loan_to_value_ratio']
    return text


raw_ltv = pd.concat([read_raw_ltv(PREPARED_DIR / 'train_features.csv'),
                     read_raw_ltv(PREPARED_DIR / 'test_features.csv')], ignore_index=True)
both = pd.concat([X_base, X_test_base], ignore_index=True)
require(len(raw_ltv) == len(both), 'LTV 文本行数不符')
require(pd.to_numeric(raw_ltv.replace('', np.nan)).equals(both['loan_to_value_ratio'])
        or np.allclose(pd.to_numeric(raw_ltv.replace('', np.nan)), both['loan_to_value_ratio'],
                       equal_nan=True), 'LTV 文本与数值不一致')

decimals = raw_ltv.str.split('.').str[1].fillna('')
precision = np.select(
    [raw_ltv.eq(''), decimals.isin(['', '0']), decimals.str.len().eq(1),
     decimals.str.len().eq(2), decimals.str.len().eq(3)],
    ['missing', 'integer', 'dec1', 'dec2', 'dec3'], default='dec4plus',
)
ltv = both['loan_to_value_ratio']
band = np.select(
    [ltv.isna(), ltv < 80, ltv == 80, ltv < 95, ltv < 97, ltv < 100, ltv == 100],
    ['missing', 'lt80', 'eq80', '80_95', '95_97', '97_100', 'eq100'], default='gt100',
)
PROGRAM_MAX_LTV = {'1': 97.0, '2': 96.5, '3': 100.0, '4': 100.0}   # 常规/FHA/VA/RHS
unknown_types = set(both['loan_type']) - set(PROGRAM_MAX_LTV)
require(not unknown_types, f'未知 loan_type：{sorted(unknown_types)}')
implied_value = safe_ratio(both['loan_amount'] * 100, ltv)
term = both['loan_term']

round2 = {
    'precision': pd.DataFrame({'ltv_precision': precision}),
    'thresholds': pd.DataFrame({
        'ltv_band': band,
        'ltv_over_program': ltv - both['loan_type'].map(PROGRAM_MAX_LTV).astype('float64'),
    }),
    'implied': pd.DataFrame({
        'implied_property_value': implied_value,
        'implied_value_ratio': safe_ratio(implied_value, both['property_value']),
    }),
    'term': pd.DataFrame({
        'term_is_360': term.eq(360).astype('float64').where(term.notna()),
        'term_years': (term / 12).round(1),
    }),
}
n_train = len(X_base)
for name, frame in round2.items():
    require(len(frame) == len(both), f'{name} 行数不符')
    BLOCKS[name] = (frame.iloc[:n_train].reset_index(drop=True),
                    frame.iloc[n_train:].reset_index(drop=True))
NEW_CATEGORICAL |= {'ltv_precision', 'ltv_band'}

train_labels_check = y.to_numpy()
for column, frame in [('ltv_precision', round2['precision']), ('ltv_band', round2['thresholds'])]:
    table = pd.crosstab(frame[column].iloc[:n_train], train_labels_check, normalize='index')
    counts = frame[column].iloc[:n_train].value_counts()
    display(table.assign(rows=counts).round(3))
for name in ['implied', 'term']:
    display(BLOCKS[name][0].describe().T.round(3))
del both
'''),
    md('''
## 6. 五折训练（可续跑）

与 10 相同的训练循环。
'''),
    code(RUN10),
    md('''
## 7. 与 G0_all 配对比较

`keep`：五折完成、平均提升 ≥ 0.0003 且至少 4/5 折优于 G0。
'''),
    code(SUMMARY10.replace("'F0_base'", "'G0_all'").replace('F0_base', 'G0_all').replace(
        'mean_delta_vs_F0', 'mean_delta_vs_G0').replace('folds_better_than_F0', 'folds_better_than_G0')),
]
save(cells12, '12_hmda_colab_gpu_feature_round2.ipynb')

# --------------------------------------------------------------------- 13
SETUP11 = cells11[2]['source']
RUN11 = cells11[10]['source']
ROUND2_CELL = cells12[10]['source']

setup13 = SETUP11.replace("EXPERIMENT_ID = 'params_x_features_v1'", "EXPERIMENT_ID = 'round2_final_v1'")
old_groups = setup13[setup13.index("GROUPS = {"):setup13.index("if RUN_ONLY is not None:")]
setup13 = setup13.replace(old_groups, """ROUND2 = ['precision', 'thresholds', 'implied', 'term']
G7 = NO_FREQ + ROUND2                      # 12 中最好的特征集（72 列）
LEAN = NO_FREQ + ['precision', 'thresholds']  # 只保留 12 中单独有效的两块（68 列）
GROUPS = {
    'M0_C11_all': dict(params=C11, blocks=ALL_BLOCKS),   # = 11 的 K2（已提交 0.92073），同轮对照
    'M1_C11_G7': dict(params=C11, blocks=G7),
    'M2_C7_G7': dict(params=C7, blocks=G7),
    'M3_C11_lean': dict(params=C11, blocks=LEAN),
}
BASELINE = 'M0_C11_all'
BLEND_GROUPS = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']   # 等权融合，五折都完成后自动计算
""")
assert "M1_C11_G7" in setup13

summary13 = '''
results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='group', values='valid_auc')
require(BASELINE in auc.columns, f'缺少对照 {BASELINE}')


def load_complete(name):
    folds = sorted(results.loc[results['group'].eq(name), 'fold'])
    if folds != [1, 2, 3, 4, 5]:
        return None
    valid = pd.concat([pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__valid.csv').assign(fold=k)
                       for k in folds]).sort_values('id', kind='stable').reset_index(drop=True)
    tests = [pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__test.csv', dtype={'id': str}) for k in folds]
    for frame in tests:
        require(frame['id'].equals(sample_submission['id']), f'{name} 测试 ID 顺序不一致')
    return valid, np.mean([t['accepted'].to_numpy() for t in tests], axis=0)


def write_submission(name, probability):
    require(np.isfinite(probability).all() and len(probability) == 45000, '提交概率无效')
    path = OUTPUT_DIR / f'submission_{name}.csv'
    pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
        path, index=False, float_format='%.12f')
    return path.name


complete = {name: load_complete(name) for name in auc.columns}
complete = {k: v for k, v in complete.items() if v is not None}
rows = []
for name in auc.columns:
    row = {'group': name, 'folds_done': int(auc[name].notna().sum()),
           'mean_auc': auc[name].mean(),
           'mean_delta_vs_M0': (auc[name] - auc[BASELINE]).mean(),
           'folds_better_than_M0': int(((auc[name] - auc[BASELINE]) > 0).sum()),
           'mean_refit_rounds': results.loc[results['group'].eq(name), 'refit_rounds'].mean()}
    if name in complete:
        valid, test_prob = complete[name]
        row['oof_auc'] = roc_auc_score(valid['accepted'], valid['prob'])
        row['submission'] = write_submission(name, test_prob)
    rows.append(row)

if all(name in complete for name in BLEND_GROUPS):
    base_valid = complete[BLEND_GROUPS[0]][0]
    for name in BLEND_GROUPS[1:]:
        require(complete[name][0]['id'].equals(base_valid['id']), '融合 OOF 的 ID 不一致')
    blend_prob = np.mean([complete[n][0]['prob'].to_numpy() for n in BLEND_GROUPS], axis=0)
    fold_auc = pd.Series({k: roc_auc_score(base_valid.loc[base_valid['fold'].eq(k), 'accepted'],
                                           blend_prob[base_valid['fold'].eq(k).to_numpy()])
                          for k in range(1, 6)})
    delta = fold_auc - auc[BASELINE]
    rows.append({'group': 'BLEND_' + '+'.join(BLEND_GROUPS), 'folds_done': 5,
                 'mean_auc': fold_auc.mean(), 'mean_delta_vs_M0': delta.mean(),
                 'folds_better_than_M0': int((delta > 0).sum()),
                 'oof_auc': roc_auc_score(base_valid['accepted'], blend_prob),
                 'submission': write_submission(
                     'blend_M1_M2_M3',
                     np.mean([complete[n][1] for n in BLEND_GROUPS], axis=0))})

table = pd.DataFrame(rows).set_index('group').sort_values('mean_auc', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(auc.sub(auc[BASELINE], axis=0).round(6))
done = table[table['folds_done'] == 5]
if not done.empty:
    best = done['mean_auc'].idxmax()
    print(f'五折平均 AUC 最高：{best}（{done.loc[best, "mean_auc"]:.6f}）→ 提交 {done.loc[best, "submission"]}')
'''

cells13 = [
    md('''
# 13 · HMDA：第二轮特征 × 最佳参数，五折确认并导出提交（Colab GPU）

12 的五折结论（快速配置，相对 68 列的 G0）：

|特征集|相对 G0|更好折数|
|---|---:|---:|
|**G7**：去掉地区频次 + 全部第二轮块（72 列）|**+0.00125**|5/5|
|LTV 申报精度（单独）|+0.00075|4/5|
|LTV 产品门槛（单独）|+0.00074|5/5|
|反推房价 / 期限（单独）|+0.0001 / +0.0002|3/5|

本 notebook 用 11 的协议（内层 15% 早停 → 外层重训；五个折模型测试概率平均作为提交）比较：

|组|参数|特征|
|---|---|---|
|`M0_C11_all`|C11|11 的 68 列（= 已提交的 K2，同轮对照）|
|`M1_C11_G7`|C11|G7（72 列）|
|`M2_C7_G7`|C7|G7|
|`M3_C11_lean`|C11|去掉频次 + 精度 + 门槛（68 列）|

三个新组五折都完成后，自动计算 **M1+M2+M3 等权融合**的五折 AUC，并写出 `submission_blend_M1_M2_M3.csv`。**选五折平均最高的提交**。

**操作**：Colab 选 **T4 GPU**，从上到下运行。每折约 7–9 分钟，全部约 2.5–3 小时；结果存 Drive `artifacts/final_gpu/<EXPERIMENT_ID>/`，断线后从头运行可续跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(setup13),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('## 6. 五折训练 + 测试预测（可续跑，与 11 相同）'),
    code(RUN11),
    md('''
## 7. 五折比较、融合与提交文件

每个完成五折的组写出 `submission_<组>.csv`；融合组写出 `submission_blend_M1_M2_M3.csv`。
'''),
    code(summary13),
]
save(cells13, '13_hmda_colab_gpu_round2_final.ipynb')

# --------------------------------------------------------------------- 14
setup14 = SETUP10.replace("EXPERIMENT_ID = 'feature_blocks_v1'", "EXPERIMENT_ID = 'feature_blocks_round3_v1'")
old_variants = setup14[setup14.index("VARIANTS = {"):setup14.index("if RUN_ONLY is not None:")]
setup14 = setup14.replace(old_variants, """NO_FREQ = ['dti', 'ltv', 'income', 'missing']
G7 = NO_FREQ + ['precision', 'thresholds', 'implied', 'term']   # 12/13 的最佳特征集（72 列）
ROUND3 = ['purpose', 'ltv_consistency', 'risk_layers', 'tract_income']
VARIANTS = {
    'H0_G7': G7,
    'H1_purpose': G7 + ['purpose'],
    'H2_ltv_consistency': G7 + ['ltv_consistency'],
    'H3_risk_layers': G7 + ['risk_layers'],
    'H4_tract_income': G7 + ['tract_income'],
    'H5_round3_all': G7 + ROUND3,
}
""")
assert 'H5_round3_all' in setup14

round3_cell = '''
both = pd.concat([X_base, X_test_base], ignore_index=True)
ltv = both['loan_to_value_ratio']
loan, value = both['loan_amount'], both['property_value']

# HMDA 公开数据中贷款额与房价都是 1 万美元区间的中点（训练集全部以 5000 结尾），
# 所以由二者算出的 LTV 只能确定一个区间；申报 LTV 落在区间外，说明用的房价口径不同。
require(((loan % 10000) == 5000).all(), '贷款额不全是 1 万区间中点，区间假设不成立')
ltv_min = 100 * safe_ratio(loan - 5000, value + 5000)
ltv_max = 100 * safe_ratio(loan + 5000, value - 5000)
outside = np.where(ltv.isna() | ltv_min.isna(), np.nan,
                   np.where(ltv < ltv_min, ltv - ltv_min, np.where(ltv > ltv_max, ltv - ltv_max, 0.0)))
consistency = np.select([ltv.isna() | ltv_min.isna(), outside == 0, outside < 0],
                        ['unknown', 'inside', 'below'], default='above')

PRODUCT_FLAGS = ['business_or_commercial_purpose', 'open-end_line_of_credit', 'reverse_mortgage',
                 'interest_only_payment', 'balloon_payment', 'negative_amortization',
                 'other_nonamortizing_features']
purpose = both[PRODUCT_FLAGS].astype(str).agg('|'.join, axis=1)

dti_upper = BLOCKS['dti'][0]['dti_upper'] if False else add_dti_bounds(both)['dti_upper']
dti_lower = add_dti_bounds(both)['dti_lower']
risk = pd.DataFrame({
    'high_ltv': ltv > 95,
    'high_dti': dti_lower >= 45,                 # 下界 ≥ 45 才确定为高 DTI
    'no_aus': both['aus-1'].eq('6'),             # aus-1 = 6：未使用自动审批系统
    'not_owner_occupied': both['occupancy_type'].ne('1'),
    'ltv_missing': ltv.isna(),
    'second_lien': both['lien_status'].eq('2'),
})
tract_median_income = both['ffiec_msa_md_median_family_income'] * both['tract_to_msa_income_percentage'] / 100

round3 = {
    'purpose': pd.DataFrame({'product_flags': 'p' + purpose}),
    'ltv_consistency': pd.DataFrame({'ltv_consistency': consistency,
                                     'ltv_outside_interval': outside}),
    'risk_layers': pd.DataFrame({'risk_layer_count': risk.sum(axis=1).astype('float64'),
                                 'risk_layer_pattern': 'r' + risk.astype(int).astype(str).agg(''.join, axis=1)}),
    'tract_income': pd.DataFrame({'income_to_tract': safe_ratio(both['income'] * 1000, tract_median_income)}),
}
n_train = len(X_base)
for name, frame in round3.items():
    require(len(frame) == len(both), f'{name} 行数不符')
    BLOCKS[name] = (frame.iloc[:n_train].reset_index(drop=True),
                    frame.iloc[n_train:].reset_index(drop=True))
NEW_CATEGORICAL |= {'product_flags', 'ltv_consistency', 'risk_layer_pattern'}

for column, block in [('product_flags', 'purpose'), ('ltv_consistency', 'ltv_consistency'),
                      ('risk_layer_count', 'risk_layers')]:
    values = BLOCKS[block][0][column]
    table = pd.crosstab(values, y.to_numpy(), normalize='index').assign(rows=values.value_counts())
    display(table.sort_values('rows', ascending=False).head(10).round(3))
display(BLOCKS['tract_income'][0].describe().T.round(3))
del both
'''.replace("dti_upper = BLOCKS['dti'][0]['dti_upper'] if False else add_dti_bounds(both)['dti_upper']\n", '')

cells14 = [
    md('''
# 14 · HMDA：第三轮特征（Colab GPU，五折）

**先做的诊断（2026-10-15，本地，只读 13 的 OOF）**：按普查区、县、MSA、州、「机构画像」组合、贷款类型×DTI、LTV×DTI 等分组检验残差离散度，几乎都与随机分组相同，说明模型在这些方向**已经没有明显缺的信息**。本轮预期提升较小（约 0–0.0008），用快速配置先筛。

基线 `H0_G7` = 12/13 的最佳特征集（72 列）。

|特征块|新增列|依据|
|---|---|---|
|`purpose`|`product_flags`（7 个产品标志拼成一个类别）|残差离散度 1.75，为本次诊断中最高（仅 16 组，证据弱）|
|`ltv_consistency`|`ltv_consistency`（区间内/低于/高于/未知）、`ltv_outside_interval`|贷款额与房价都是 1 万区间中点，由它们只能确定 LTV 的区间；申报值落在区间外说明房价口径不同（细化已有效的 `ltv_gap`）|
|`risk_layers`|`risk_layer_count`、`risk_layer_pattern`|审批中常见的「风险叠加」：高 LTV、高 DTI、未用 AUS、非自住、LTV 缺失、次级留置|
|`tract_income`|`income_to_tract`|收入相对所在普查区的中位收入（区别于 10 中相对 MSA 的比值）|

**协议**同 10/12：快速配置，固定五折，与 `H0_G7` 同折配对。考虑到 GPU 运行间约 0.0005 的差异，**五折平均 ≥ +0.0005 且 ≥ 4/5 折**才保留。

**操作**：Colab 选 **T4 GPU**，从上到下运行；6 个变体 × 5 折 × 约 2 分钟 ≈ 60–70 分钟；结果存 Drive `artifacts/feature_ablation/<EXPERIMENT_ID>/`，可续跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(setup14),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('## 6. 第三轮特征块'),
    code(round3_cell),
    md('## 7. 五折训练（可续跑，与 10 相同）'),
    code(RUN10),
    md('''
## 8. 与 H0_G7 配对比较

`keep`：五折完成、平均提升 ≥ 0.0005 且至少 4/5 折优于 H0。
'''),
    code(SUMMARY10.replace("'F0_base'", "'H0_G7'").replace('F0_base', 'H0_G7').replace(
        'mean_delta_vs_F0', 'mean_delta_vs_H0').replace('folds_better_than_F0', 'folds_better_than_H0').replace(
        ">= 0.0003)", ">= 0.0005)")),
]
save(cells14, '14_hmda_colab_gpu_feature_round3.ipynb')

# --------------------------------------------------------------------- 15
setup15 = setup13.replace(
    "BLEND_GROUPS = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']   # 等权融合，五折都完成后自动计算\n",
    """CAT_BASES = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']
SEEDS_EXTRA = [2027, 31415]          # 13 已用 seed 42；这里只补跑新种子
for base_name in CAT_BASES:
    for extra_seed in SEEDS_EXTRA:
        base_group = GROUPS[base_name]
        GROUPS[f'{base_name}_s{extra_seed}'] = dict(
            params=dict(base_group['params'], random_seed=extra_seed), blocks=base_group['blocks'])
RUN_ONLY = [f'{b}_s{s}' for b in CAT_BASES for s in SEEDS_EXTRA]   # M0–M3 已在 13 完成

LGBM_BLOCKS = G7
LGBM_BASE = dict(
    objective='binary', learning_rate=0.03, n_estimators=MAX_ITERATIONS,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, cat_l2=10.0, cat_smooth=20.0, min_data_per_group=50,
    max_cat_threshold=64, random_state=SEED, n_jobs=os.cpu_count() or 2, verbose=-1,
)
LGBM_CONFIGS = {                      # 不搜索，只取两个容量；融合时两者等权平均
    'L1_lgbm_leaves31': dict(num_leaves=31, min_child_samples=50),
    'L2_lgbm_leaves127': dict(num_leaves=127, min_child_samples=100),
}
""")
assert 'LGBM_CONFIGS' in setup15 and "EXPERIMENT_ID = 'round2_final_v1'" in setup15

lgbm_cell = '''
try:
    import lightgbm as lgb
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm'])
    import lightgbm as lgb
print('LightGBM', lgb.__version__)

lgb_train, lgb_test, lgb_cats = variant_frames(LGBM_BLOCKS)
for column in lgb_cats:               # 训练+测试的类别并集作为固定词表（不涉及标签）
    dtype = pd.CategoricalDtype(sorted(set(lgb_train[column]) | set(lgb_test[column])))
    lgb_train[column] = lgb_train[column].astype(dtype)
    lgb_test[column] = lgb_test[column].astype(dtype)
lgb_train.columns = [c.replace('-', '_') for c in lgb_train.columns]
lgb_test.columns = lgb_train.columns


def run_lgbm(name, config, fold):
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    started = time.perf_counter()
    inner_fit, inner_stop = train_test_split(
        fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED, stratify=y.iloc[fit_idx])
    probe = lgb.LGBMClassifier(**LGBM_BASE, **config)
    probe.fit(lgb_train.iloc[inner_fit], y.iloc[inner_fit],
              eval_set=[(lgb_train.iloc[inner_stop], y.iloc[inner_stop])], eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(OD_WAIT, verbose=False)])
    best_iteration = int(probe.best_iteration_)
    rounds = int(round(best_iteration / (1 - INNER_STOP_FRACTION)))
    model = lgb.LGBMClassifier(**dict(LGBM_BASE, n_estimators=rounds), **config)
    model.fit(lgb_train.iloc[fit_idx], y.iloc[fit_idx])
    valid_prob = checked_probabilities(model, lgb_train.iloc[valid_idx])
    test_prob = checked_probabilities(model, lgb_test)
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'{name}__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': test_prob}).to_csv(
        f'{stem}__test.csv', index=False, float_format='%.12f')
    result = {
        'group': name, 'fold': fold,
        'definition': {'model': 'lightgbm', 'config': config, 'blocks': LGBM_BLOCKS},
        'feature_count': lgb_train.shape[1], 'inner_best_iteration': best_iteration,
        'early_stopped': best_iteration < MAX_ITERATIONS, 'refit_rounds': rounds,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'lightgbm': lgb.__version__,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)
    return result


for fold in FOLDS:
    for name, config in LGBM_CONFIGS.items():
        path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            require(saved['definition']['config'] == config, f'{name} 的定义已改变；请改名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        result = run_lgbm(name, config, fold)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  轮数={result["refit_rounds"]}  '
              f'{result["seconds"]/60:.1f} 分钟', flush=True)
'''

blend15 = '''
results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
done = {g for g, part in results.groupby('group') if sorted(part['fold']) == [1, 2, 3, 4, 5]}
print('五折完成的组：', sorted(done))


def load(name):
    valid = pd.concat([pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__valid.csv').assign(fold=k)
                       for k in range(1, 6)]).sort_values('id', kind='stable').reset_index(drop=True)
    tests = [pd.read_csv(OUTPUT_DIR / f'{name}__fold{k}__test.csv', dtype={'id': str}) for k in range(1, 6)]
    for frame in tests:
        require(frame['id'].equals(sample_submission['id']), f'{name} 测试 ID 顺序不一致')
    return valid, np.mean([t['accepted'].to_numpy() for t in tests], axis=0)


loaded = {name: load(name) for name in sorted(done)}
reference = loaded[CAT_BASES[0]][0][['id', 'accepted', 'fold']]
for name, (valid, _) in loaded.items():
    require(valid['id'].equals(reference['id']), f'{name} OOF 的 ID 与参照不一致')


def mean_of(names):
    return (np.mean([loaded[n][0]['prob'].to_numpy() for n in names], axis=0),
            np.mean([loaded[n][1] for n in names], axis=0))


def fold_aucs(probability):
    return pd.Series({k: roc_auc_score(reference.loc[reference['fold'].eq(k), 'accepted'],
                                       probability[reference['fold'].eq(k).to_numpy()])
                      for k in range(1, 6)})


blends = {}
require(all(b in done for b in CAT_BASES), '13 的 M1–M3 结果缺失')
blends['B13_seed42'] = mean_of(CAT_BASES)                       # = 已提交的 0.92190
per_base = [mean_of([b] + [f'{b}_s{s}' for s in SEEDS_EXTRA if f'{b}_s{s}' in done]) for b in CAT_BASES]
n_seeds = {b: 1 + sum(f'{b}_s{s}' in done for s in SEEDS_EXTRA) for b in CAT_BASES}
seed_tag = 'B_seeds_' + '_'.join(str(n_seeds[b]) for b in CAT_BASES)
blends[seed_tag] = (np.mean([v for v, _ in per_base], axis=0), np.mean([t for _, t in per_base], axis=0))
lgbm_names = [n for n in LGBM_CONFIGS if n in done]
if lgbm_names:
    blends['LGBM_avg'] = mean_of(lgbm_names)
    cat_valid, cat_test = blends[seed_tag]
    lgb_valid, lgb_test_prob = blends['LGBM_avg']
    for weight in [0.1, 0.2, 0.3]:                               # 只看三个预设权重，不做搜索
        blends[f'{seed_tag}+LGBM_w{int(weight * 100)}'] = (
            (1 - weight) * cat_valid + weight * lgb_valid,
            (1 - weight) * cat_test + weight * lgb_test_prob)
    cat_corr = np.corrcoef(cat_valid, lgb_valid)[0, 1]
    print(f'CatBoost 融合 与 LightGBM 的 OOF 相关系数：{cat_corr:.4f}')

ref_folds = fold_aucs(blends['B13_seed42'][0])
rows = []
for name, (valid_prob, test_prob) in blends.items():
    folds = fold_aucs(valid_prob)
    require(np.isfinite(test_prob).all() and len(test_prob) == 45000, f'{name} 测试概率无效')
    path = OUTPUT_DIR / f'submission_{name}.csv'
    pd.DataFrame({'id': sample_submission['id'], 'accepted': test_prob}).to_csv(
        path, index=False, float_format='%.12f')
    rows.append({'blend': name, 'mean_auc': folds.mean(),
                 'delta_vs_B13': (folds - ref_folds).mean(),
                 'folds_better_than_B13': int(((folds - ref_folds) > 0).sum()),
                 'oof_auc': roc_auc_score(reference['accepted'], valid_prob),
                 'submission': path.name})
table = pd.DataFrame(rows).set_index('blend').sort_values('mean_auc', ascending=False)
table.to_csv(OUTPUT_DIR / 'blend_summary.csv')
display(table.round(6))
singles = results.groupby('group')['valid_auc'].agg(['count', 'mean']).round(6)
display(singles.sort_values('mean', ascending=False))
best = table['mean_auc'].idxmax()
print(f'五折平均最高：{best}（{table.loc[best, "mean_auc"]:.6f}，相对已提交 {table.loc[best, "delta_vs_B13"]:+.6f}）'
      f' → {table.loc[best, "submission"]}')
'''

cells15 = [
    md('''
# 15 · HMDA：最终集成——LightGBM 互补 + CatBoost 多种子（Colab GPU）

特征已到饱和：14 的第三轮四个特征块都没有达到保留标准（最好 +0.00035，3/5 折），13 的残差诊断也显示没有明显缺失的信息。所以最后阶段改为**提升最终提交的稳定性与多样性**。

本 notebook 与 13 共用同一个结果目录 `artifacts/final_gpu/round2_final_v1/`，**直接复用 13 已完成的 M0–M3**，只补跑：

1. **LightGBM**（CPU，两个容量，G7 特征）：另一种学习机制，检查与 CatBoost 的互补性。每折约 2–5 分钟，先跑。
2. **CatBoost 多种子**：M1/M2/M3 各加 seed 2027、31415（外层五折、内层早停划分都不变，只换模型种子）。6 组 × 5 折 × 约 8 分钟 ≈ 4 小时，可分多次续跑。

最后一节计算几种**预先定好**的融合并全部写出提交文件：

|名称|内容|
|---|---|
|`B13_seed42`|13 的 M1+M2+M3（已提交 0.92190），参照|
|`B_seeds_a_b_c`|M1/M2/M3 各自先对已完成的种子取平均，再等权；数字为各自种子数|
|`LGBM_avg`|两个 LightGBM 等权|
|`B_seeds…+LGBM_w10/20/30`|CatBoost 多种子融合与 LightGBM 按 90/10、80/20、70/30 加权|

LightGBM 的三个权重是在同一份 OOF 上比较的，挑最好的会有轻微的选择偏差；只有比 `B13_seed42` 明显更好（约 +0.0003 以上且 ≥ 4/5 折）才值得替换。**最后一节可以随时运行**：只用已经完成五折的组。

**操作**：Colab 选 **T4 GPU**，从上到下运行。断线后从头运行会跳过已完成的部分。
'''),
    md('## 1. 环境、GPU 与配置（结果目录与 13 相同）'),
    code(setup15),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('''
## 6. LightGBM 五折（CPU，可续跑）

类别列转为 pandas category，词表为训练+测试类别的并集。早停与 CatBoost 相同：外层训练部分内 15% 早停，再按 `b / 0.85` 轮在整个外层训练部分重训。
'''),
    code(lgbm_cell),
    md('''
## 7. CatBoost 新种子五折（GPU，可续跑，与 11/13 相同的训练函数）

已完成的 M0–M3 会被 `RUN_ONLY` 排除；只训练带 `_s2027`、`_s31415` 后缀的组。这一节最耗时，可以中途停止，先去第 8 节看部分结果。
'''),
    code(RUN11),
    md('## 8. 预设融合与提交文件（随时可运行）'),
    code(blend15),
]
save(cells15, '15_hmda_colab_final_ensemble.ipynb')

# --------------------------------------------------------------------- 16
setup16 = SETUP_COMMON + GPU_CHECK + '''
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

EXPERIMENT_ID = 'tabm_mini_v1'
FOLDS = [1]                 # 先只跑第 1 折试水；值得继续再改为 [1, 2, 3, 4, 5]
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'nn_models' / EXPERIMENT_ID
CATBOOST_DIR = PROJECT_DIR / 'artifacts' / 'final_gpu' / 'round2_final_v1'   # 13/15 的结果
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NO_FREQ = ['dti', 'ltv', 'income', 'missing']
G7 = NO_FREQ + ['precision', 'thresholds', 'implied', 'term']   # 与最终 CatBoost 相同的 72 列
NN_CONFIG = dict(
    k=32,                   # TabM 的「集成成员」数：32 个共享大部分权重的 MLP
    width=512, depth=3, dropout=0.1,
    cat_dim=8,              # 每个类别列的嵌入维度
    ple_bins=16,            # 数值列分段线性编码的分段数
    rare_min_count=5,       # 外层训练部分出现少于 5 次的类别并入「未知」
    lr=1e-3, weight_decay=1e-4, batch_size=512,
    max_epochs=60, patience=8,
    inner_stop_fraction=0.15,
)
BLEND_WEIGHTS = [0.1, 0.2, 0.3]
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f'项目：{PROJECT_DIR}')
print(f'设备：{DEVICE}（{GPU_NAME}）；torch {torch.__version__}；折：{FOLDS}；输出：{OUTPUT_DIR}')
'''

nn_prep_cell = '''
nn_train, nn_test, nn_cats = variant_frames(G7)
nn_nums = [c for c in nn_train.columns if c not in nn_cats]
print(f'特征：{nn_train.shape[1]} 列 = {len(nn_nums)} 数值 + {len(nn_cats)} 类别')


class FoldPreprocessor:
    """只在外层训练部分上拟合：分位数正态化、分段线性编码 (PLE)、缺失标记、类别词表。"""

    def __init__(self, bins, rare_min_count):
        self.bins = bins
        self.rare_min_count = rare_min_count

    def fit(self, frame):
        num = frame[nn_nums].to_numpy(dtype='float64')
        self.quantile = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                            subsample=10**9, random_state=SEED)
        self.quantile.fit(num)                      # sklearn 会忽略 NaN
        self.edges = []
        for j in range(num.shape[1]):
            values = num[:, j][~np.isnan(num[:, j])]
            edges = np.unique(np.quantile(values, np.linspace(0, 1, self.bins + 1)))
            self.edges.append(edges if len(edges) >= 2 else np.array([values.min(), values.min() + 1]))
        self.has_missing = np.isnan(num).any(axis=0)
        self.vocab = []
        for column in nn_cats:
            counts = frame[column].value_counts()
            kept = counts.index[counts >= self.rare_min_count]
            self.vocab.append({value: i + 1 for i, value in enumerate(kept)})   # 0 = 未知/稀有
        self.cardinalities = [len(v) + 1 for v in self.vocab]
        return self

    def transform(self, frame):
        num = frame[nn_nums].to_numpy(dtype='float64')
        missing = np.isnan(num)
        normal = np.nan_to_num(np.clip(self.quantile.transform(num), -5, 5), nan=0.0)
        ple_parts = []
        for j, edges in enumerate(self.edges):
            x = num[:, j:j + 1]
            left, right = edges[:-1][None, :], edges[1:][None, :]
            part = np.clip((x - left) / np.maximum(right - left, 1e-12), 0, 1)
            part = np.nan_to_num(part, nan=0.0)
            ple_parts.append(part)
        dense = np.concatenate([normal] + ple_parts + [missing[:, self.has_missing]], axis=1)
        cats = np.stack([frame[c].map(v).fillna(0).to_numpy(dtype='int64')
                         for c, v in zip(nn_cats, self.vocab)], axis=1)
        return dense.astype('float32'), cats


class TabMMini(nn.Module):
    """TabM-mini：k 个成员共享同一个 MLP，只在输入端各有一个随机 ±1 的缩放向量；预测取 k 个成员概率的平均。"""

    def __init__(self, d_dense, cardinalities, k, width, depth, dropout, cat_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, cat_dim) for c in cardinalities])
        d_in = d_dense + cat_dim * len(cardinalities)
        self.adapter = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(d_in if i == 0 else width, width), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head_weight = nn.Parameter(torch.randn(k, width) / width ** 0.5)
        self.head_bias = nn.Parameter(torch.zeros(k))

    def forward(self, dense, cats):
        parts = [dense] + [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(parts, dim=1)                       # (B, D)
        x = x[:, None, :] * self.adapter[None]            # (B, k, D)
        h = self.backbone(x)                              # (B, k, width)
        return (h * self.head_weight).sum(-1) + self.head_bias   # (B, k) logits


def predict(model, dense, cats, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(dense), batch):
            d = torch.from_numpy(dense[start:start + batch]).to(DEVICE)
            c = torch.from_numpy(cats[start:start + batch]).to(DEVICE)
            out.append(torch.sigmoid(model(d, c)).mean(1).cpu().numpy())
    return np.concatenate(out).astype('float64')


def train_fold(fold):
    cfg = NN_CONFIG
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    inner_fit, inner_stop = train_test_split(
        fit_idx, test_size=cfg['inner_stop_fraction'], random_state=SEED, stratify=y.iloc[fit_idx])
    prep = FoldPreprocessor(cfg['ple_bins'], cfg['rare_min_count']).fit(nn_train.iloc[fit_idx])
    data = {name: prep.transform(frame) for name, frame in [
        ('fit', nn_train.iloc[inner_fit]), ('stop', nn_train.iloc[inner_stop]),
        ('valid', nn_train.iloc[valid_idx]), ('test', nn_test)]}
    y_fit = torch.from_numpy(y.iloc[inner_fit].to_numpy(dtype='float32')).to(DEVICE)
    y_stop = y.iloc[inner_stop].to_numpy()

    torch.manual_seed(SEED + fold)
    model = TabMMini(data['fit'][0].shape[1], prep.cardinalities, cfg['k'], cfg['width'],
                     cfg['depth'], cfg['dropout'], cfg['cat_dim']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    dense_fit = torch.from_numpy(data['fit'][0]).to(DEVICE)
    cats_fit = torch.from_numpy(data['fit'][1]).to(DEVICE)
    best_auc, best_state, best_epoch, waited = -1.0, None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(dense_fit), device=DEVICE)
        for start in range(0, len(order), cfg['batch_size']):
            batch = order[start:start + cfg['batch_size']]
            logits = model(dense_fit[batch], cats_fit[batch])
            loss = loss_fn(logits, y_fit[batch][:, None].expand_as(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        stop_auc = roc_auc_score(y_stop, predict(model, *data['stop']))
        if stop_auc > best_auc:
            best_auc, best_epoch, waited = stop_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        print(f'  fold {fold} epoch {epoch:2d}: 内层早停集 AUC={stop_auc:.5f}'
              f'{"  *" if waited == 0 else ""}', flush=True)
        if waited >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    valid_prob = predict(model, *data['valid'])
    test_prob = predict(model, *data['test'])
    require(np.isfinite(valid_prob).all() and np.isfinite(test_prob).all(), '神经网络预测含非有限值')
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'TabMMini__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': test_prob}).to_csv(
        f'{stem}__test.csv', index=False, float_format='%.12f')
    result = {
        'group': 'TabMMini', 'fold': fold, 'config': cfg, 'blocks': G7,
        'dense_dim': int(data['fit'][0].shape[1]), 'categorical_columns': len(nn_cats),
        'best_epoch': best_epoch, 'inner_stop_auc': best_auc,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'device': DEVICE,
        'torch': torch.__version__, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)
    del model, dense_fit, cats_fit
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return result
'''

nn_run_cell = '''
fixed_config(OUTPUT_DIR, {
    'purpose': 'TabM-mini (PyTorch) on G7 features; inner 15% early stop, no outer refit',
    'seed': SEED, 'nn_config': NN_CONFIG, 'blocks': G7, 'data_hashes': DATA_HASHES,
})
for fold in FOLDS:
    path = OUTPUT_DIR / f'TabMMini__fold{fold}.json'
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf-8'))
        print(f'TabMMini fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
        continue
    result = train_fold(fold)
    print(f'TabMMini fold {fold}: AUC={result["valid_auc"]:.6f}  最佳 epoch={result["best_epoch"]}  '
          f'{result["seconds"]/60:.1f} 分钟', flush=True)
'''

nn_blend_cell = '''
nn_done = sorted(json.loads(p.read_text(encoding='utf-8'))['fold']
                 for p in OUTPUT_DIR.glob('TabMMini__fold?.json'))
require(nn_done, '神经网络还没有完成的折')

CAT_BASES = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']
SEED_SUFFIXES = ['', '_s2027', '_s31415']
cat_groups = [b + s for b in CAT_BASES for s in SEED_SUFFIXES]
missing = [g for g in cat_groups for k in range(1, 6)
           if not (CATBOOST_DIR / f'{g}__fold{k}__valid.csv').is_file()]
require(not missing, f'缺少 15 的 CatBoost 结果，例如 {missing[:3]}；请先完成 15')


def catboost_fold(fold):
    """15 的最终方案 B_seeds：每个基础组先对 3 个种子平均，再三组等权。"""
    per_base = []
    for base in CAT_BASES:
        frames = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{fold}__valid.csv') for s in SEED_SUFFIXES]
        for f in frames[1:]:
            require(f['id'].equals(frames[0]['id']), 'CatBoost OOF 行顺序不一致')
        per_base.append(np.mean([f['prob'].to_numpy() for f in frames], axis=0))
    return frames[0][['id', 'accepted']].assign(cat=np.mean(per_base, axis=0))


rows = []
for fold in nn_done:
    nn_valid = pd.read_csv(OUTPUT_DIR / f'TabMMini__fold{fold}__valid.csv')
    merged = catboost_fold(fold).merge(nn_valid[['id', 'prob']], on='id', validate='one_to_one')
    require(len(merged) == len(nn_valid), '神经网络与 CatBoost 的验证行不一致')
    row = {'fold': fold,
           'catboost_auc': roc_auc_score(merged['accepted'], merged['cat']),
           'tabm_auc': roc_auc_score(merged['accepted'], merged['prob']),
           'corr': np.corrcoef(merged['cat'], merged['prob'])[0, 1]}
    for w in BLEND_WEIGHTS:
        row[f'blend_w{int(w * 100)}'] = roc_auc_score(merged['accepted'],
                                                     (1 - w) * merged['cat'] + w * merged['prob'])
    rows.append(row)
table = pd.DataFrame(rows).set_index('fold')
for w in BLEND_WEIGHTS:
    table[f'delta_w{int(w * 100)}'] = table[f'blend_w{int(w * 100)}'] - table['catboost_auc']
display(table.round(6))
summary = table.mean().to_frame('mean').T
display(summary.round(6))
table.to_csv(OUTPUT_DIR / 'blend_by_fold.csv')

gap = summary['catboost_auc'].iloc[0] - summary['tabm_auc'].iloc[0]
best_w = max(BLEND_WEIGHTS, key=lambda w: summary[f'delta_w{int(w * 100)}'].iloc[0])
best_delta = summary[f'delta_w{int(w * 100) if False else int(best_w * 100)}'].iloc[0]
print(f'TabM 单独比 CatBoost 低 {gap:.5f}；与 CatBoost 的相关系数 {summary["corr"].iloc[0]:.4f}；'
      f'最好的融合权重 {best_w}：{best_delta:+.6f}')
if len(nn_done) < 5:
    verdict = ('值得跑完五折：把 FOLDS 改为 [1, 2, 3, 4, 5] 后从头运行（第 1 折会自动跳过）。'
               if best_delta >= 0.0003 else
               '第 1 折上融合提升不足 0.0003，建议到此为止，保持 15 的 B_seeds 作为最终提交。')
    print('判断：' + verdict)
else:
    better = int((table[f'delta_w{int(best_w * 100)}'] > 0).sum())
    print(f'五折：权重 {best_w} 时 {better}/5 折优于 CatBoost。')
    cat_test = []
    for base in CAT_BASES:
        tests = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{k}__test.csv', dtype={'id': str})
                 for s in SEED_SUFFIXES for k in range(1, 6)]
        for t in tests:
            require(t['id'].equals(sample_submission['id']), 'CatBoost 测试 ID 顺序不一致')
        cat_test.append(np.mean([t['accepted'].to_numpy() for t in tests], axis=0))
    cat_test = np.mean(cat_test, axis=0)
    nn_tests = [pd.read_csv(OUTPUT_DIR / f'TabMMini__fold{k}__test.csv', dtype={'id': str})
                for k in range(1, 6)]
    for t in nn_tests:
        require(t['id'].equals(sample_submission['id']), 'TabM 测试 ID 顺序不一致')
    nn_test = np.mean([t['accepted'].to_numpy() for t in nn_tests], axis=0)
    for w in BLEND_WEIGHTS:
        path = OUTPUT_DIR / f'submission_Bseeds_TabM_w{int(w * 100)}.csv'
        pd.DataFrame({'id': sample_submission['id'],
                      'accepted': (1 - w) * cat_test + w * nn_test}).to_csv(
            path, index=False, float_format='%.12f')
        print('写出', path.name)
    print('只有当某个权重五折平均 ≥ +0.0003 且 ≥ 4/5 折更好时，才替换 15 的 B_seeds。')
'''.replace("summary[f'delta_w{int(w * 100) if False else int(best_w * 100)}']", "summary[f'delta_w{int(best_w * 100)}']")

cells16 = [
    md('''
# 16 · HMDA：神经网络（TabM-mini）与 CatBoost 融合（Colab GPU）

**目的**：现有最终提交（15 的 `B_seeds`）全部是 CatBoost，彼此预测相关 > 0.98，互相平均能抵消的错误有限；LightGBM 同样是树模型（相关 0.979）且单独太弱，所以没有帮助。神经网络的学习方式完全不同，**如果它单独不太弱（约 0.915 以上）**，与 CatBoost 融合才有机会互补。

**模型**：TabM-mini（Gorishniy 等，2024 的简化版，PyTorch 自己实现，不依赖外部包）。32 个「成员」共享同一个 3 层 MLP，每个成员只在输入端有自己的 ±1 缩放向量；预测取 32 个成员概率的平均，相当于一次训练得到一个小集成。

**输入**（与最终 CatBoost 相同的 G7，72 列），全部只在外层训练部分上拟合：
- 数值列：分位数正态化 + 16 段分段线性编码（PLE）+ 缺失标记；
- 类别列：词表 + 8 维嵌入，出现少于 5 次或未见过的类别归入「未知」。

**协议**：固定 seed=42 五折；外层训练部分再留出 15% 做早停（按 AUC，耐心 8 个 epoch），**不再在整个外层训练部分重训**（神经网络的轮数外推不可靠）。

**流程**
1. 默认 `FOLDS = [1]`：只跑第 1 折（预计 5–15 分钟），最后一节直接给出与 CatBoost（15 的 B_seeds）在同一折上的融合效果和建议。
2. 若第 1 折融合提升 ≥ 0.0003，把 `FOLDS` 改成 `[1, 2, 3, 4, 5]` 从头运行（第 1 折自动跳过），最后一节会写出三个权重的提交文件。
3. 否则到此为止，最终提交不变。

需要 15 已经完成（读取 `artifacts/final_gpu/round2_final_v1/` 中 9 个 CatBoost 组的五折预测）。Colab 选 **T4 GPU**。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(setup16),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('''
## 6. 折内预处理与 TabM-mini 模型

预处理器在每个外层折的训练部分上重新拟合，验证和测试只做变换。
'''),
    code(nn_prep_cell),
    md('## 7. 训练（每折结束即保存，可续跑）'),
    code(nn_run_cell),
    md('''
## 8. 与 CatBoost（15 的 B_seeds）在同一折上比较与融合

只用已完成的折。`delta_wXX` 为「(1−w)·CatBoost + w·TabM」相对 CatBoost 的 AUC 变化。
'''),
    code(nn_blend_cell),
]
save(cells16, '16_hmda_colab_gpu_tabm_blend.ipynb')

# --------------------------------------------------------------------- 17
setup17 = setup13.replace("EXPERIMENT_ID = 'round2_final_v1'", "EXPERIMENT_ID = 'full_refit_v1'").replace(
    "BLEND_GROUPS = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']   # 等权融合，五折都完成后自动计算\n",
    """CATBOOST_DIR = PROJECT_DIR / 'artifacts' / 'final_gpu' / 'round2_final_v1'   # 13/15 的折模型结果
CAT_BASES = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']
SEED_SUFFIXES = {42: '', 2027: '_s2027', 31415: '_s31415'}
LC_GROUP = 'M3_C11_lean'       # 学习曲线只用五折最好的单组
LC_FRACTION = 0.8              # 外层训练部分的 80%（67,200 行）对比 100%（84,000 行）
FULL_ROUNDS_SCALE = 105000 / 84000   # 全量轮数 = 该组五折重训轮数均值 × 数据量之比（与 b/0.85 同一规则）
RUN_LEARNING_CURVE = True
RUN_FULL_REFIT = True
""")
assert 'FULL_ROUNDS_SCALE' in setup17

train17 = '''
fixed_config(OUTPUT_DIR, {
    'purpose': 'learning curve (80% vs 100% of outer-train) + full-data refit of M1-M3 x 3 seeds',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'max_iterations': MAX_ITERATIONS, 'od_wait': OD_WAIT,
    'lc_group': LC_GROUP, 'lc_fraction': LC_FRACTION, 'full_rounds_scale': FULL_ROUNDS_SCALE,
    'gpu_base': {k: v for k, v in GPU_BASE.items() if k != 'thread_count'},
    'data_hashes': DATA_HASHES,
})


def model_params(spec, iterations, seed):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations, random_seed=seed)


def early_stop_refit(spec, train, cats, idx, seed):
    """与 11/13 完全相同的协议：idx 内留 15% 早停 → 在整个 idx 上按 b/0.85 轮重训。"""
    inner_fit, inner_stop = train_test_split(idx, test_size=INNER_STOP_FRACTION,
                                             random_state=SEED, stratify=y.iloc[idx])
    probe = catboost.CatBoostClassifier(**model_params(spec, MAX_ITERATIONS, seed),
                                        eval_metric='Logloss', od_type='Iter',
                                        od_wait=OD_WAIT, use_best_model=True)
    probe.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
              eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
    best = int(probe.get_best_iteration()) + 1
    rounds = int(round(best / (1 - INNER_STOP_FRACTION)))
    del probe
    gc.collect()
    model = catboost.CatBoostClassifier(**model_params(spec, rounds, seed), use_best_model=False)
    model.fit(train.iloc[idx], y.iloc[idx], cat_features=cats)
    require(int(model.tree_count_) == rounds, '重训轮数不符')
    return model, best, rounds
'''

lc17 = '''
if RUN_LEARNING_CURVE:
    lc_spec = GROUPS[LC_GROUP]
    lc_train, _, lc_cats = variant_frames(lc_spec['blocks'])
    for fold in [1, 2, 3, 4, 5]:
        fit_idx = np.flatnonzero(fold_assignment != fold)
        valid_idx = np.flatnonzero(fold_assignment == fold)
        sub_idx, _ = train_test_split(fit_idx, train_size=LC_FRACTION, random_state=SEED + fold,
                                      stratify=y.iloc[fit_idx])
        for name, idx in [('LC_sub80', np.sort(sub_idx)), ('LC_full', fit_idx)]:
            path = OUTPUT_DIR / f'{name}__fold{fold}.json'
            if path.is_file():
                saved = json.loads(path.read_text(encoding='utf-8'))
                print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
                continue
            started = time.perf_counter()
            model, best, rounds = early_stop_refit(lc_spec['params'], lc_train, lc_cats, idx, SEED)
            probability = checked_probabilities(model, lc_train.iloc[valid_idx])
            result = {'name': name, 'fold': fold, 'group': LC_GROUP, 'train_rows': int(len(idx)),
                      'inner_best_iteration': best, 'refit_rounds': rounds,
                      'valid_auc': roc_auc_score(y.iloc[valid_idx], probability),
                      'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                      'finished_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json(path, result)
            print(f'{name} fold {fold}: {len(idx):,} 行  AUC={result["valid_auc"]:.6f}  '
                  f'轮数={rounds}  {result["seconds"]/60:.1f} 分钟', flush=True)
            del model
            gc.collect()

lc = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT_DIR.glob('LC_*__fold?.json')])
if not lc.empty:
    curve = lc.pivot(index='fold', columns='name', values='valid_auc')
    if {'LC_sub80', 'LC_full'} <= set(curve.columns):
        curve['gain_84k_vs_67k'] = curve['LC_full'] - curve['LC_sub80']
    reference = {k: json.loads((CATBOOST_DIR / f'{LC_GROUP}__fold{k}.json').read_text(encoding='utf-8'))['valid_auc']
                 for k in curve.index}
    curve['M3_in_13'] = pd.Series(reference)
    display(curve.round(6))
    if 'gain_84k_vs_67k' in curve:
        done = curve['gain_84k_vs_67k'].dropna()
        print(f'训练数据 ×1.25（67k → 84k）：平均 AUC {done.mean():+.6f}，{int((done > 0).sum())}/{len(done)} 折为正。'
              f' 全量重训（84k → 105k）是同样的 ×1.25，预期收益与此同量级。')
'''

full17 = '''
def group_rounds(group):
    rounds = [json.loads((CATBOOST_DIR / f'{group}__fold{k}.json').read_text(encoding='utf-8'))['refit_rounds']
              for k in range(1, 6)]
    return float(np.mean(rounds))


if RUN_FULL_REFIT:
    for base in CAT_BASES:
        spec = GROUPS[base]
        train, test, cats = variant_frames(spec['blocks'])
        for seed, suffix in SEED_SUFFIXES.items():
            name = f'FULL_{base}_seed{seed}'
            path = OUTPUT_DIR / f'{name}.json'
            if path.is_file():
                print(f'{name}: 已完成，跳过')
                continue
            rounds = int(round(group_rounds(base + suffix) * FULL_ROUNDS_SCALE))
            started = time.perf_counter()
            model = catboost.CatBoostClassifier(**model_params(spec['params'], rounds, seed),
                                                use_best_model=False)
            model.fit(train, y, cat_features=cats)
            require(int(model.tree_count_) == rounds, '全量模型轮数不符')
            probability = checked_probabilities(model, test)
            pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': probability}).to_csv(
                OUTPUT_DIR / f'{name}__test.csv', index=False, float_format='%.12f')
            write_json(path, {'name': name, 'base': base, 'seed': seed, 'train_rows': len(train),
                              'rounds': rounds, 'seconds': time.perf_counter() - started,
                              'gpu_name': GPU_NAME,
                              'finished_at_utc': datetime.now(timezone.utc).isoformat()})
            print(f'{name}: {rounds} 轮  {(time.perf_counter() - started)/60:.1f} 分钟', flush=True)
            del model
            gc.collect()
'''

submit17 = '''
def fold_average_bseeds():
    per_base = []
    for base in CAT_BASES:
        tests = [pd.read_csv(CATBOOST_DIR / f'{base}{s}__fold{k}__test.csv', dtype={'id': str})
                 for s in SEED_SUFFIXES.values() for k in range(1, 6)]
        for t in tests:
            require(t['id'].equals(sample_submission['id']), '折模型测试 ID 顺序不一致')
        per_base.append(np.mean([t['accepted'].to_numpy() for t in tests], axis=0))
    return np.mean(per_base, axis=0)


full_paths = {(b, s): OUTPUT_DIR / f'FULL_{b}_seed{s}__test.csv' for b in CAT_BASES for s in SEED_SUFFIXES}
missing = [p.name for p in full_paths.values() if not p.is_file()]
if missing:
    print(f'全量模型还缺 {len(missing)} 个（例如 {missing[0]}）；完成后再运行本格。')
else:
    per_base = []
    for base in CAT_BASES:
        frames = [pd.read_csv(full_paths[(base, s)], dtype={'id': str}) for s in SEED_SUFFIXES]
        for f in frames:
            require(f['id'].equals(sample_submission['id']), '全量模型测试 ID 顺序不一致')
        per_base.append(np.mean([f['accepted'].to_numpy() for f in frames], axis=0))
    full = np.mean(per_base, axis=0)
    fold_avg = fold_average_bseeds()
    print(f'全量融合 与 折平均 B_seeds 的测试预测相关系数：{np.corrcoef(full, fold_avg)[0, 1]:.5f}')
    for name, probability in [('full_Bseeds', full), ('hybrid_50_50', 0.5 * full + 0.5 * fold_avg)]:
        require(np.isfinite(probability).all() and len(probability) == 45000, f'{name} 概率无效')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
            path, index=False, float_format='%.12f')
        print('写出', path.name)
    if 'curve' in globals() and 'gain_84k_vs_67k' in curve and curve['gain_84k_vs_67k'].notna().sum() == 5:
        gain = curve['gain_84k_vs_67k']
        if gain.mean() >= 0.0005 and (gain > 0).sum() >= 4:
            advice = '学习曲线支持全量重训：最终主选 submission_hybrid_50_50.csv，备选保留 15 的 B_seeds。'
        elif gain.mean() >= 0.0002:
            advice = '学习曲线收益偏小：hybrid 可以作为备选，主选仍为 15 的 B_seeds。'
        else:
            advice = '学习曲线没有显示数据量收益：保持 15 的 B_seeds，不使用全量模型。'
        print('判断：' + advice)
'''

cells17 = [
    md('''
# 17 · HMDA：学习曲线 + 全量重训（Colab GPU）

**问题**：目前所有提交都是「五个折模型的测试预测平均」，每个折模型只见过 **80% 的训练数据（84,000 行）**，没有任何一个模型用上全部 105,000 行。

**本 notebook 回答两件事**

1. **多 25% 数据值多少？**（学习曲线）同一次运行、同一折、同一协议，M3（五折最好的单组）分别在外层训练部分的 80%（67,200 行）和 100%（84,000 行）上训练，比较外层验证 AUC。67k→84k 与 84k→105k 都是 ×1.25，前者的收益就是全量重训收益的直接估计。`LC_full` 同时可与 13 中的 M3 逐折对照，检查可复现性。
2. **把它用起来。**（全量重训）M1/M2/M3 × 3 个种子（42、2027、31415）在全部 105,000 行上训练，轮数 = 该组五折重训轮数均值 × 1.25（与全项目 b/0.85 的同一规则）。写出：
   - `submission_full_Bseeds.csv`：9 个全量模型（先按组平均种子，再三组等权）；
   - `submission_hybrid_50_50.csv`：全量与 15 的折平均 B_seeds 各半，兼顾「更多数据」与「折模型的多样性」。

全量模型没有可用的验证集，是否采用**只依据第 1 部分的学习曲线**：平均 ≥ +0.0005 且 ≥ 4/5 折为正，才把 hybrid 作为最终主选。

**操作**：Colab 选 **T4 GPU**，从上到下运行。学习曲线 10 次训练约 70–80 分钟，全量 9 次约 90 分钟；每次训练后立即保存，断线后从头运行可续跑。可以先只看第 6 节学习曲线的结果再决定是否继续（把 `RUN_FULL_REFIT=False` 即可跳过全量部分）。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(setup17),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('## 6. 训练函数（与 11/13 相同的早停与重训协议）'),
    code(train17),
    md('''
## 7. 学习曲线：80% 对 100% 外层训练数据

`gain_84k_vs_67k` 为同一折上 100% 减 80% 的 AUC；`M3_in_13` 是 13 中同一配置的结果，`LC_full` 应与它基本一致。
'''),
    code(lc17),
    md('## 8. 全量重训：M1/M2/M3 × 3 个种子（可续跑）'),
    code(full17),
    md('## 9. 写出提交文件与建议'),
    code(submit17),
]
save(cells17, '17_hmda_colab_gpu_full_refit.ipynb')

# --------------------------------------------------------------------- 18
setup18 = setup17.replace("EXPERIMENT_ID = 'full_refit_v1'", "EXPERIMENT_ID = 'pseudo_label_v1'").replace(
    """LC_GROUP = 'M3_C11_lean'       # 学习曲线只用五折最好的单组
LC_FRACTION = 0.8              # 外层训练部分的 80%（67,200 行）对比 100%（84,000 行）
FULL_ROUNDS_SCALE = 105000 / 84000   # 全量轮数 = 该组五折重训轮数均值 × 数据量之比（与 b/0.85 同一规则）
RUN_LEARNING_CURVE = True
RUN_FULL_REFIT = True
""",
    """FULL_DIR = PROJECT_DIR / 'artifacts' / 'final_gpu' / 'full_refit_v1'          # 17 的全量模型
PL_GROUP = 'M3_C11_lean'       # 验证阶段只用五折最好的单组
PL_WEIGHT = 1.0                # 每个伪标签行的样本权重（真实行为 1）
RUN_FINAL_PL = 'auto'          # 'auto'：验证通过（≥ +0.0003 且 ≥ 4/5 折）才做全量；也可设 True / False
""")
assert 'PL_WEIGHT' in setup18 and "'pseudo_label_v1'" in setup18

train18 = train17.replace(
    "    'lc_group': LC_GROUP, 'lc_fraction': LC_FRACTION, 'full_rounds_scale': FULL_ROUNDS_SCALE,\n",
    "    'pl_group': PL_GROUP, 'pl_weight': PL_WEIGHT,\n").replace(
    "'purpose': 'learning curve (80% vs 100% of outer-train) + full-data refit of M1-M3 x 3 seeds',",
    "'purpose': 'soft pseudo-labelling of test rows (CrossEntropy), fold-honest validation + full-data students',").replace(
    """def model_params(spec, iterations, seed):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations, random_seed=seed)""",
    """def model_params(spec, iterations, seed, loss='Logloss'):
    params = {k: v for k, v in spec.items() if k not in {'early_stop', 'iterations'}}
    return dict(GPU_BASE, **params, iterations=iterations, random_seed=seed, loss_function=loss)""") + '''

def pseudo_label_refit(spec, train, test, cats, idx, pseudo, seed):
    """真实行（idx）+ 全部测试行（软标签 pseudo）一起训练。早停集只取真实行的 15%，
    重训轮数按有效样本量（真实行 + 权重 × 伪标签行）等比放大。"""
    inner_fit, inner_stop = train_test_split(idx, test_size=INNER_STOP_FRACTION,
                                             random_state=SEED, stratify=y.iloc[idx])

    def stacked(real_idx):
        frame = pd.concat([train.iloc[real_idx], test], ignore_index=True)
        target = np.concatenate([y.iloc[real_idx].to_numpy(dtype='float64'), pseudo])
        weight = np.concatenate([np.ones(len(real_idx)), np.full(len(test), PL_WEIGHT)])
        return frame, target, weight

    X_fit, y_fit, w_fit = stacked(inner_fit)
    probe = catboost.CatBoostClassifier(**model_params(spec, MAX_ITERATIONS, seed, loss='CrossEntropy'),
                                        eval_metric='CrossEntropy', od_type='Iter',
                                        od_wait=OD_WAIT, use_best_model=True)
    probe.fit(X_fit, y_fit, sample_weight=w_fit, cat_features=cats,
              eval_set=(train.iloc[inner_stop], y.iloc[inner_stop].astype('float64')))
    best = int(probe.get_best_iteration()) + 1
    effective_inner = len(inner_fit) + PL_WEIGHT * len(test)
    effective_full = len(idx) + PL_WEIGHT * len(test)
    rounds = int(round(best * effective_full / effective_inner))
    del probe, X_fit, y_fit, w_fit
    gc.collect()
    X_all, y_all, w_all = stacked(idx)
    model = catboost.CatBoostClassifier(**model_params(spec, rounds, seed, loss='CrossEntropy'),
                                        use_best_model=False)
    model.fit(X_all, y_all, sample_weight=w_all, cat_features=cats)
    require(int(model.tree_count_) == rounds, '伪标签模型轮数不符')
    return model, best, rounds


def mean_test(paths):
    frames = [pd.read_csv(p, dtype={'id': str}) for p in paths]
    for f in frames:
        require(f['id'].equals(sample_submission['id']), f'测试 ID 顺序不一致：{paths[0]}')
    return np.mean([f['accepted'].to_numpy() for f in frames], axis=0)


def teacher_for_fold(fold):
    """第 k 折的老师 = 13/15 中只在第 k 折外层训练部分上训练的 9 个模型对测试集的平均预测。"""
    return mean_test([CATBOOST_DIR / f'{b}{s}__fold{fold}__test.csv'
                      for b in CAT_BASES for s in SEED_SUFFIXES.values()])


def teacher_valid(fold):
    frames = [pd.read_csv(CATBOOST_DIR / f'{b}{s}__fold{fold}__valid.csv')
              for b in CAT_BASES for s in SEED_SUFFIXES.values()]
    for f in frames[1:]:
        require(f['id'].equals(frames[0]['id']), '老师 OOF 行顺序不一致')
    return frames[0][['id']].assign(teacher=np.mean([f['prob'].to_numpy() for f in frames], axis=0))
'''

validate18 = '''
pl_spec = GROUPS[PL_GROUP]
pl_train, pl_test, pl_cats = variant_frames(pl_spec['blocks'])
for fold in [1, 2, 3, 4, 5]:
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    pseudo = teacher_for_fold(fold)
    for name in ['A_base', 'A_pseudo']:
        path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        started = time.perf_counter()
        if name == 'A_base':
            model, best, rounds = early_stop_refit(pl_spec['params'], pl_train, pl_cats, fit_idx, SEED)
        else:
            model, best, rounds = pseudo_label_refit(pl_spec['params'], pl_train, pl_test, pl_cats,
                                                     fit_idx, pseudo, SEED)
        probability = checked_probabilities(model, pl_train.iloc[valid_idx])
        pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': y.iloc[valid_idx].to_numpy(),
                      'prob': probability}).to_csv(OUTPUT_DIR / f'{name}__fold{fold}__valid.csv',
                                                   index=False, float_format='%.12f')
        result = {'name': name, 'fold': fold, 'group': PL_GROUP, 'pl_weight': PL_WEIGHT,
                  'inner_best_iteration': best, 'refit_rounds': rounds,
                  'valid_auc': roc_auc_score(y.iloc[valid_idx], probability),
                  'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                  'finished_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(path, result)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  轮数={rounds}  '
              f'{result["seconds"]/60:.1f} 分钟', flush=True)
        del model
        gc.collect()

res = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT_DIR.glob('A_*__fold?.json')])
table = res.pivot(index='fold', columns='name', values='valid_auc')
pl_gain = None
if {'A_base', 'A_pseudo'} <= set(table.columns):
    table['gain_pseudo'] = table['A_pseudo'] - table['A_base']
    blend_rows = {}
    for fold in table.index:
        if not (OUTPUT_DIR / f'A_pseudo__fold{fold}__valid.csv').is_file():
            continue
        student = pd.read_csv(OUTPUT_DIR / f'A_pseudo__fold{fold}__valid.csv')
        merged = teacher_valid(fold).merge(student, on='id', validate='one_to_one')
        auc_teacher = roc_auc_score(merged['accepted'], merged['teacher'])
        auc_mix = roc_auc_score(merged['accepted'], 0.5 * merged['teacher'] + 0.5 * merged['prob'])
        blend_rows[fold] = {'teacher_9models': auc_teacher, 'mix_gain_vs_teacher': auc_mix - auc_teacher}
    table = table.join(pd.DataFrame(blend_rows).T)
    rounds = res.pivot(index='fold', columns='name', values='refit_rounds')
    table['rounds_ratio'] = rounds['A_pseudo'] / rounds['A_base']
    done = table['gain_pseudo'].dropna()
    pl_gain = (done.mean(), int((done > 0).sum()), len(done))
display(table.round(6))
if pl_gain:
    print(f'伪标签（单模型对单模型）：平均 {pl_gain[0]:+.6f}，{pl_gain[1]}/{pl_gain[2]} 折为正。'
          f' mix_gain_vs_teacher 表示学生与 9 模型老师各半后相对老师本身的变化（参考）。')
'''

final18 = '''
complete = pl_gain is not None and pl_gain[2] == 5
passed = complete and pl_gain[0] >= 0.0003 and pl_gain[1] >= 4
do_final = passed if RUN_FINAL_PL == 'auto' else bool(RUN_FINAL_PL)
if not do_final:
    print('跳过全量伪标签训练：' + ('验证未完成。' if not complete else
          f'验证未通过（平均 {pl_gain[0]:+.6f}，{pl_gain[1]}/5 折）。最终提交保持 17 的 full_Bseeds。'))
else:
    ratio = float(table['rounds_ratio'].mean())
    full_teacher = mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for b in CAT_BASES for s in SEED_SUFFIXES])
    all_idx = np.arange(len(y))
    for base in CAT_BASES:
        spec = GROUPS[base]
        train, test, cats = variant_frames(spec['blocks'])
        for seed in SEED_SUFFIXES:
            name = f'FULLPL_{base}_seed{seed}'
            path = OUTPUT_DIR / f'{name}.json'
            if path.is_file():
                print(f'{name}: 已完成，跳过')
                continue
            full_info = json.loads((FULL_DIR / f'FULL_{base}_seed{seed}.json').read_text(encoding='utf-8'))
            rounds = int(round(full_info['rounds'] * ratio))
            started = time.perf_counter()
            X_all = pd.concat([train, test], ignore_index=True)
            y_all = np.concatenate([y.to_numpy(dtype='float64'), full_teacher])
            w_all = np.concatenate([np.ones(len(train)), np.full(len(test), PL_WEIGHT)])
            model = catboost.CatBoostClassifier(**model_params(spec['params'], rounds, seed, loss='CrossEntropy'),
                                                use_best_model=False)
            model.fit(X_all, y_all, sample_weight=w_all, cat_features=cats)
            require(int(model.tree_count_) == rounds, '全量伪标签模型轮数不符')
            probability = checked_probabilities(model, test)
            pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': probability}).to_csv(
                OUTPUT_DIR / f'{name}__test.csv', index=False, float_format='%.12f')
            write_json(path, {'name': name, 'base': base, 'seed': seed, 'rounds': rounds,
                              'rounds_ratio': ratio, 'seconds': time.perf_counter() - started,
                              'gpu_name': GPU_NAME, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
            print(f'{name}: {rounds} 轮  {(time.perf_counter() - started)/60:.1f} 分钟', flush=True)
            del model, X_all
            gc.collect()

    students = np.mean([mean_test([OUTPUT_DIR / f'FULLPL_{b}_seed{s}__test.csv' for s in SEED_SUFFIXES])
                        for b in CAT_BASES], axis=0)
    full_bseeds = np.mean([mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for s in SEED_SUFFIXES])
                           for b in CAT_BASES], axis=0)
    print(f'伪标签全量融合 与 17 的 full_Bseeds 测试预测相关系数：{np.corrcoef(students, full_bseeds)[0, 1]:.5f}')
    for name, probability in [('full_PL', students), ('full_PL_mix', 0.5 * students + 0.5 * full_bseeds)]:
        require(np.isfinite(probability).all() and len(probability) == 45000, f'{name} 概率无效')
        path = OUTPUT_DIR / f'submission_{name}.csv'
        pd.DataFrame({'id': sample_submission['id'], 'accepted': probability}).to_csv(
            path, index=False, float_format='%.12f')
        print('写出', path.name)
    print('建议：验证通过时 submission_full_PL.csv 作主选，17 的 submission_full_Bseeds.csv 作备选。')
'''

cells18 = [
    md('''
# 18 · HMDA：测试集软伪标签（Colab GPU）

**为什么做**：17 的学习曲线显示，同一模型的训练数据从 67k 增加到 84k（×1.25），外层验证 AUC 平均 **+0.0012，5/5 折为正**；全量重训在公开榜上也从 0.92197 升到 0.92301。模型目前**受数据量限制**。真实标签已经全部用完，测试集还有 45,000 行未标注数据。

**做法（软伪标签 / 知识蒸馏）**：用现有最好的模型给测试集每一行一个批准概率，把这些行连同概率作为目标（CatBoost `CrossEntropy` 损失接受 [0,1] 的目标）加入训练。权重 `PL_WEIGHT = 1`。

**诚实的验证（第 1 部分，M3 单组、seed 42）**：对每个外层折 k，
- 老师 = 13/15 中**只在第 k 折外层训练部分上训练**的 9 个模型（M1/M2/M3 × 3 种子）对测试集的平均预测；
- `A_base`：与 11/13/17 完全相同的协议，只用真实行；
- `A_pseudo`：真实行 + 45,000 个软标签测试行；早停集只取真实行的 15%，重训轮数按有效样本量等比放大；
- 两者在**同一次运行、同一折**上比较外层验证 AUC。第 k 折验证标签始终不参与。另列出「学生与 9 模型老师各半」相对老师本身的变化作参考。

**全量（第 2 部分，验证通过才自动执行）**：老师 = 17 的 9 个全量模型；M1/M2/M3 × 3 种子在 105,000 真实行 + 45,000 软标签行上训练，轮数 = 17 的全量轮数 × 第 1 部分的轮数比例。写出 `submission_full_PL.csv` 与 `submission_full_PL_mix.csv`（与 17 的 full_Bseeds 各半）。

**规则提醒**：这一步使用了测试集的特征（不使用任何标签）。Kaggle 类比赛通常允许，但请确认本课程比赛的规则。

**操作**：Colab 选 **T4 GPU**，从上到下运行。第 1 部分 10 次训练约 1.5–2 小时；第 2 部分 9 次约 1–1.5 小时。断线后从头运行可续跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(setup18),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('## 6. 训练函数'),
    code(train18),
    md('''
## 7. 第 1 部分：逐折诚实验证（可续跑）

`gain_pseudo` = A_pseudo − A_base（同折）。`rounds_ratio` 用于第 2 部分的轮数。
'''),
    code(validate18),
    md('## 8. 第 2 部分：全量伪标签训练与提交文件'),
    code(final18),
]
save(cells18, '18_hmda_colab_gpu_pseudo_label.ipynb')

# --------------------------------------------------------------------- 19
setup19 = SETUP_COMMON + GPU_CHECK + '''
EXPERIMENT_ID = 'data_efficiency_screen_v1'
FOLDS = [1, 2, 3, 4, 5]
RUN_ONLY = None                 # 例如 ['V0_base', 'V1_ordered']；None 表示全部
INNER_STOP_FRACTION = 0.15
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'gpu_tuning' / EXPERIMENT_ID

NO_FREQ = ['dti', 'ltv', 'income', 'missing']
LEAN = NO_FREQ + ['precision', 'thresholds']        # M3 的 68 列特征
DEMO_DUPLICATES = [                                 # 与 derived_* 重复的人口统计原始编码与「观察」标记
    'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
    'applicant_race-1', 'co-applicant_race-1',
    'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed',
    'applicant_race_observed', 'co-applicant_race_observed',
    'applicant_sex_observed', 'co-applicant_sex_observed',
]
SCREEN_BASE = dict(              # 接近 M3（C11）但学习率加倍以加快筛选；外层验证只评分
    depth=6, learning_rate=0.06, l2_leaf_reg=10,
    bootstrap_type='Bernoulli', subsample=0.8,
    loss_function='Logloss', eval_metric='Logloss', nan_mode='Min', random_seed=SEED,
    task_type='GPU', devices='0', thread_count=4,
    iterations=6000, od_type='Iter', od_wait=200, use_best_model=True,
    allow_writing_files=False, verbose=False,
)
VARIANTS = {
    'V0_base': {},
    'V1_ordered': {'boosting_type': 'Ordered'},            # 超过 5 万行时 GPU 默认 Plain，从未试过
    'V2_newtonL2': {'score_function': 'NewtonL2'},         # 默认 Cosine
    'V3_depthwise': {'grow_policy': 'Depthwise', 'depth': 7, 'min_data_in_leaf': 50},
    'V4_lossguide': {'grow_policy': 'Lossguide', 'depth': 10, 'max_leaves': 48,
                     'min_data_in_leaf': 50, 'score_function': 'NewtonL2'},
    'V5_poisson': {'bootstrap_type': 'Poisson', 'subsample': 0.8},   # 仅 GPU 支持
    'V6_sub066': {'subsample': 0.66},                      # 更强的行抽样
    'V7_border254': {'border_count': 254},                 # GPU 默认 128 个数值分箱
    'V8_ctr1': {'max_ctr_complexity': 1},                  # 不做类别组合，降低方差
    'V9_drop_demo': {'drop': DEMO_DUPLICATES},             # 删去 10 个重复的人口统计列
}
if RUN_ONLY is not None:
    unknown = set(RUN_ONLY) - set(VARIANTS)
    if unknown:
        raise ValueError(f'RUN_ONLY 中有未知变体：{sorted(unknown)}')
print(f'项目：{PROJECT_DIR}')
print(f'GPU：{GPU_NAME}；CatBoost {catboost.__version__}；折：{FOLDS}；输出：{OUTPUT_DIR}')
'''

preflight19 = '''
lean_train, lean_test, lean_cats = variant_frames(LEAN)


def frames_for(spec):
    drop = spec.get('drop', [])
    unknown = set(drop) - set(lean_train.columns)
    require(not unknown, f'要删除的列不存在：{sorted(unknown)}')
    columns = [c for c in lean_train.columns if c not in drop]
    return lean_train[columns], [c for c in lean_cats if c not in drop]


def params_for(spec, **extra):
    params = dict(SCREEN_BASE)
    params.update({k: v for k, v in spec.items() if k != 'drop'})
    params.update(extra)
    return params


# 预检：每个变体在 3,000 行上训练 20 轮，几秒内暴露 GPU 不支持的参数组合，避免跑了一小时才报错
rng = np.random.default_rng(SEED)
probe_rows = rng.choice(len(y), 3000, replace=False)
failed = {}
for name, spec in VARIANTS.items():
    train, cats = frames_for(spec)
    try:
        model = catboost.CatBoostClassifier(**params_for(spec, iterations=20))
        model.fit(train.iloc[probe_rows[:2500]], y.iloc[probe_rows[:2500]], cat_features=cats,
                  eval_set=(train.iloc[probe_rows[2500:]], y.iloc[probe_rows[2500:]]))
        checked_probabilities(model, train.iloc[probe_rows[2500:]])
        print(f'预检通过：{name}（{train.shape[1]} 列）')
    except Exception as exc:
        failed[name] = repr(exc)
        print(f'预检失败，跳过：{name} → {exc!r}')
require('V0_base' not in failed, '基线 V0_base 预检失败，请检查环境')
'''

run19 = '''
fixed_config(OUTPUT_DIR, {
    'purpose': 'data-efficiency screen on M3 features; inner 15% early stop, score probe model on outer fold',
    'seed': SEED, 'inner_stop_fraction': INNER_STOP_FRACTION,
    'screen_base': {k: v for k, v in SCREEN_BASE.items() if k != 'thread_count'},
    'features': LEAN, 'data_hashes': DATA_HASHES,
})
selected = {n: s for n, s in VARIANTS.items()
            if (RUN_ONLY is None or n in RUN_ONLY) and n not in failed}
for fold in FOLDS:
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION,
                                             random_state=SEED, stratify=y.iloc[fit_idx])
    for name, spec in selected.items():
        path = OUTPUT_DIR / f'{name}__fold{fold}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            require(saved['spec'] == spec, f'{name} 的定义已改变；请改变体名')
            print(f'{name} fold {fold}: 已完成 AUC={saved["valid_auc"]:.6f}，跳过')
            continue
        train, cats = frames_for(spec)
        started = time.perf_counter()
        model = catboost.CatBoostClassifier(**params_for(spec))
        model.fit(train.iloc[inner_fit], y.iloc[inner_fit], cat_features=cats,
                  eval_set=(train.iloc[inner_stop], y.iloc[inner_stop]))
        best = int(model.get_best_iteration()) + 1
        probability = checked_probabilities(model, train.iloc[valid_idx])
        actual = y.iloc[valid_idx]
        pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                      'prob': probability}).to_csv(OUTPUT_DIR / f'{name}__fold{fold}__valid.csv',
                                                   index=False, float_format='%.12f')
        result = {'variant': name, 'fold': fold, 'spec': spec, 'feature_count': train.shape[1],
                  'best_iteration': best,
                  'hit_iteration_limit': best >= SCREEN_BASE['iterations'] - SCREEN_BASE['od_wait'],
                  'valid_auc': roc_auc_score(actual, probability),
                  'valid_logloss': log_loss(actual, probability, labels=[0, 1]),
                  'seconds': time.perf_counter() - started, 'gpu_name': GPU_NAME,
                  'finished_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(path, result)
        print(f'{name} fold {fold}: AUC={result["valid_auc"]:.6f}  最佳轮数={best}  '
              f'{result["seconds"]/60:.1f} 分钟', flush=True)
        del model
        gc.collect()
'''

summary19 = '''
results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                        for p in sorted(OUTPUT_DIR.glob('*__fold?.json'))])
require(not results.empty, '还没有完成的结果')
auc = results.pivot(index='fold', columns='variant', values='valid_auc')
require('V0_base' in auc.columns, '缺少基线 V0_base')
delta = auc.sub(auc['V0_base'], axis=0)
table = pd.DataFrame({
    'folds_done': auc.notna().sum(),
    'mean_auc': auc.mean(),
    'mean_delta_vs_V0': delta.mean(),
    'folds_better_than_V0': (delta > 0).sum(),
    'mean_best_iteration': results.groupby('variant')['best_iteration'].mean(),
    'total_minutes': results.groupby('variant')['seconds'].sum() / 60,
})
table['keep'] = ((table['folds_done'] == 5) & (table['mean_delta_vs_V0'] >= 0.0005)
                 & (table['folds_better_than_V0'] >= 4))
table.loc['V0_base', 'keep'] = False
table = table.sort_values('mean_delta_vs_V0', ascending=False)
table.to_csv(OUTPUT_DIR / 'summary.csv')
display(table.round(6))
display(delta.round(6))
if failed:
    print('预检失败、未运行的变体：', failed)
kept = table.index[table['keep']].tolist()
print('达到保留标准（≥ +0.0005 且 ≥ 4/5 折）的变体：', kept if kept else '无')
'''

cells19 = [
    md('''
# 19 · HMDA：让模型更「省数据」——CatBoost 训练方式筛选（Colab GPU，五折）

**为什么做**：17 的学习曲线显示模型**受数据量限制**（训练数据 ×1.25 → AUC +0.0012，5/5 折）；18 显示伪标签几乎无效（+0.00008，3/5 折）——新的**真实**标签已经没有了。剩下的方向是让模型在同样的数据上**更稳、方差更小**。我们之前所有模型都只用了 GPU 默认的训练方式（对称树、Plain 提升、Cosine 打分、128 个分箱），以下选项**从未试过**：

|变体|改动|依据|
|---|---|---|
|`V1_ordered`|`boosting_type='Ordered'`|CatBoost 为小数据设计的有序提升，减少梯度估计的偏差；超过 5 万行时 GPU 默认关闭|
|`V2_newtonL2`|`score_function='NewtonL2'`|用二阶信息选择分裂（默认 Cosine）|
|`V3_depthwise`|逐层生长，深度 7，叶子至少 50 行|非对称树，叶子大小有下限|
|`V4_lossguide`|按损失逐叶生长，最多 48 个叶子，叶子至少 50 行|类似 LightGBM 的生长方式，但保留 CatBoost 的类别编码|
|`V5_poisson`|Poisson 抽样（仅 GPU）|另一种随机化|
|`V6_sub066`|行抽样 0.66（现为 0.8）|更强的随机化|
|`V7_border254`|254 个数值分箱（GPU 默认 128）|更细的数值切分|
|`V8_ctr1`|`max_ctr_complexity=1`|不做类别组合，降低类别编码的方差|
|`V9_drop_demo`|删去 10 个与 `derived_*` 重复的人口统计列|减少噪声列|

**协议**：特征 = M3 的 68 列；基线 `V0_base` ≈ M3（depth 6、L2=10、Bernoulli 0.8），学习率提高到 0.06 以加快筛选。每折在外层训练部分内留出 15% 早停，**直接用早停模型给外层验证打分**（不重训，所有变体相同，比较公平且省一半时间）。同一次运行、同一折配对比较。

**判读**：五折平均 ≥ +0.0005 且 ≥ 4/5 折才算有效；有效的变体之后用完整协议确认并做全量训练（20）。

**操作**：Colab 选 **T4 GPU**，从上到下运行。开头的「预检」会在一分钟内发现 GPU 不支持的参数并自动跳过该变体。预计约 1.5–2.5 小时（有序提升较慢）；每次训练后立即保存，断线后从头运行可续跑。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(setup19),
    md('## 2. 回读并核对 01 的数据契约（与 04 相同）'),
    code(DATA_CELL + HELPERS),
    md('## 3. DTI 上下界（与 04 相同）'),
    code(DTI_CELL),
    md('## 4. 第一轮特征块（与 10 相同）'),
    code(BLOCKS_CELL),
    md('## 5. 第二轮特征块（与 12 相同）'),
    code(ROUND2_CELL),
    md('## 6. 特征选择与参数预检'),
    code(preflight19),
    md('## 7. 五折筛选（可续跑）'),
    code(run19),
    md('''
## 8. 与 V0_base 配对比较

`keep`：五折完成、平均提升 ≥ 0.0005 且至少 4/5 折优于 V0。
'''),
    code(summary19),
]
save(cells19, '19_hmda_colab_gpu_data_efficiency.ipynb')
