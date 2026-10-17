"""生成项目的 5 本 notebook。

    python tools/build_notebooks.py <输出目录>

notebook 中经过 Colab 验证的训练代码保存在 tools/cells/*.py，本脚本只负责组装：
统一的环境设置、说明文字、只读的结果展示，以及放在训练开关后面的原始训练代码。
项目根目录里的 notebook 带有 Colab 运行输出，修改时先输出到其他目录，确认后再复制。
"""
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

TOOLS = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else TOOLS.parent


def cell(name):
    return (TOOLS / 'cells' / f'{name}.py').read_text(encoding='utf-8').strip('\n')


def md(text):
    # 先去掉缩进再插入 notebook 对照表：表格行没有缩进，提前插入会让 dedent 失效、正文被渲染成代码块
    text = textwrap.dedent(text).strip()
    return nbf.v4.new_markdown_cell(text.replace('{NOTEBOOK_MAP}', NOTEBOOK_MAP.strip()))


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip('\n'))


def raw_code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))


def gated(flag, body, gpu=True, catboost=True, setup=''):
    """训练代码只在开关打开时执行；关闭时本节只读取已保存的结果。"""
    lines = [f'if {flag}:']
    if catboost:
        lines.append('    ensure_catboost()')
    if gpu:
        lines.append('    ensure_gpu()')
    if setup:
        lines.append(textwrap.indent(textwrap.dedent(setup).strip('\n'), '    '))
    lines.append(textwrap.indent(body.strip('\n'), '    '))
    lines.append('else:')
    lines.append(f'    print("训练开关关闭（{flag} = False）：本节只读取 artifacts/ 中已保存的结果。")')
    return nbf.v4.new_code_cell('\n'.join(lines))


def save(cells, name):
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


# ----------------------------------------------------------------------------- 共用单元格
SETUP = '''
# 环境、路径与通用函数（5 本 notebook 相同）
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
    candidates.append(Path('/content/drive/MyDrive/Colab Notebooks/demandes-de-pret-immobilier-aux-etats-unis'))
    here = Path.cwd().resolve()
    candidates += [here, *here.parents]
    for candidate in candidates:
        if (candidate / 'data' / 'train.csv').is_file() and (candidate / 'data' / 'test.csv').is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError('找不到项目目录（应包含 data/train.csv 与 data/test.csv）；请设置环境变量 HMDA_PROJECT_DIR')


PROJECT_DIR = find_project_dir()
DATA_DIR = PROJECT_DIR / 'data'                                        # 原始 CSV（只读）
ARTIFACTS = PROJECT_DIR / 'artifacts'                                  # 每本 notebook 的结果写入同名子目录
PREPARED_DIR = ARTIFACTS / '01_data_exploration'                       # 01：固定数据接口
FEATURE_DIR = ARTIFACTS / '02_feature_engineering' / 'feature_store'   # 02：特征库
SEED = 42
CATBOOST_VERSION = '1.2.10'
GPU_NAME = None
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})


def ensure_catboost():
    """只在训练时调用：安装并导入固定版本的 CatBoost。"""
    global catboost
    try:
        installed = importlib.metadata.version('catboost')
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != CATBOOST_VERSION:
        if 'catboost' in sys.modules:
            raise RuntimeError('当前运行时已导入另一版 CatBoost；请重启运行时后从头运行')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', f'catboost=={CATBOOST_VERSION}'])
        importlib.invalidate_caches()
    import catboost as catboost_module
    if catboost_module.__version__ != CATBOOST_VERSION:
        raise RuntimeError(f'需要 CatBoost {CATBOOST_VERSION}，实际为 {catboost_module.__version__}')
    catboost = catboost_module
    return catboost_module


def ensure_gpu():
    """只在训练时调用：确认 Colab 分配了 NVIDIA GPU。"""
    global GPU_NAME
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('训练需要 GPU：Colab「更改运行时类型」选 T4 GPU 后从头运行')
    probe = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                           text=True, capture_output=True, check=False)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError('NVIDIA GPU 不可用：请选择 T4 GPU 后从头运行')
    GPU_NAME = probe.stdout.strip().splitlines()[0]
    return GPU_NAME


print(f'项目：{PROJECT_DIR}')
print(f'Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}')
'''

TRAIN_NOTE = '''
**训练开关**：默认全部为 `False`，只读取 `artifacts/` 中已保存的结果并生成表格和图（Colab 普通 CPU 运行时即可，几分钟跑完）。把某一项改为 `True` 并选择 **T4 GPU** 运行时，会按原协议重新训练对应部分；每个模型训练完立即写入 `artifacts/`，断线后从头运行会跳过已完成的部分。
'''

DATA = cell('data_contract') + '\n\n' + cell('helpers')

FEATURE_LOADER = '''
# 读取 02 生成的特征库（只在重新训练时真正需要；只读结果时缺少也不影响）
def load_feature_store():
    meta_path = FEATURE_DIR / 'feature_store.json'
    require(meta_path.is_file(), '缺少特征库：请先运行 02_feature_engineering.ipynb')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    require(meta['data_hashes'] == DATA_HASHES, '特征库与 01 的固定数据接口不一致：请重跑 02')
    stores = {}
    for part, expected_rows in [('train', len(X_base)), ('test', len(X_test_base))]:
        path = FEATURE_DIR / f'blocks_{part}.parquet'
        require(sha256_file(path) == meta['files'][path.name], f'{path.name} 哈希不符：请重跑 02')
        frame = pd.read_parquet(path)
        require(len(frame) == expected_rows, f'{path.name} 行数不符')
        for column in frame.columns:
            frame[column] = frame[column].astype(object if column in meta['categorical'] else 'float64')
        stores[part] = frame
    blocks = {name: (stores['train'][cols].reset_index(drop=True), stores['test'][cols].reset_index(drop=True))
              for name, cols in meta['blocks'].items()}
    return blocks, set(meta['categorical']), meta


def variant_frames(blocks):
    """原始 56 列 + 指定的特征块；与 02 中的定义相同。"""
    train_parts = [X_base] + [BLOCKS[b][0] for b in blocks]
    test_parts = [X_test_base] + [BLOCKS[b][1] for b in blocks]
    train = pd.concat(train_parts, axis=1)
    test = pd.concat(test_parts, axis=1)
    require(train.columns.is_unique and train.columns.equals(test.columns), '变体列不一致')
    require(train.iloc[:, :len(FEATURE_COLUMNS)].equals(X_base), '原始 56 列被修改')
    cats = CATEGORICAL_COLUMNS + [c for c in train.columns if c in NEW_CATEGORICAL]
    return train, test, cats


if (FEATURE_DIR / 'feature_store.json').is_file():
    BLOCKS, NEW_CATEGORICAL, FEATURE_META = load_feature_store()
    print(f"特征库：{len(BLOCKS)} 个特征块、{sum(len(c) for c in FEATURE_META['blocks'].values())} 列；"
          f"特征集 {list(FEATURE_META['feature_sets'])}")
else:
    BLOCKS, NEW_CATEGORICAL, FEATURE_META = None, set(), None
    print('尚未生成特征库（02）：只读取已保存结果不受影响；需要重新训练时请先运行 02。')
'''

RESULT_HELPERS = '''
# 只读的结果工具：读取每个（模型/变体, 折）保存的 JSON，做同折配对比较
def load_runs(directory, pattern='*__fold?.json'):
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    require(paths, f'{directory} 中没有结果文件：请确认 artifacts 已从 Drive 同步，或打开对应的训练开关')
    return pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in paths])


def paired_table(runs, key, baseline, min_gain=0.0005, min_folds=4):
    auc = runs.pivot(index='fold', columns=key, values='valid_auc')
    require(baseline in auc.columns, f'缺少基线 {baseline}')
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
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel(f'AUC change vs {baseline} (x1e-4); bar = 5-fold mean, dots = single folds')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
'''

OOF_HELPERS = '''
# 读取五折验证预测（OOF）：每个文件是一个外层折的验证行
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
        require(f['id'].equals(sample_submission['id']), f'测试 ID 顺序不一致：{Path(paths[0]).name}')
    return np.mean([f['accepted'].to_numpy() for f in frames], axis=0)


CV_DIR = ARTIFACTS / '04_final_model' / 'cv_models'      # 五折模型（M0 与 M1–M3 × 3 个种子）
FULL_DIR = ARTIFACTS / '04_final_model' / 'full_refit'   # 学习曲线与全量模型
CAT_BASE_GROUPS = ['M1_C11_G7', 'M2_C7_G7', 'M3_C11_lean']
SEED_TAGS = {42: '', 2027: '_s2027', 31415: '_s31415'}
'''

NOTEBOOK_MAP = '''
| notebook | 内容 |
|---|---|
| **01_data_exploration** | 原始数据读取、审计、探索分析、数据质量检查、固定五折 |
| **02_feature_engineering** | 全部特征的定义、三轮特征消融、保存特征库 |
| **03_model_selection** | 基线模型（逻辑回归、XGBoost、CatBoost、AutoGluon）、线性模型的数值变换、CatBoost 调参与训练方式筛选 |
| **04_final_model** | 最终五折模型、多种子、学习曲线、全量重训与最终提交 |
| **05_error_analysis** | 残差诊断、LightGBM 与神经网络、跨实验融合、伪标签、公开榜噪声与运行间差异 |

原始数据在 `data/`；每本 notebook 的结果保存在 `artifacts/` 下与它同名的目录。
'''

# ============================================================================= 01
nb01 = [
    md('''
    # 01 · 数据读取与分析

    **任务**：预测美国住房贷款申请（HMDA 数据）是否被批准，输出 `accepted = 1` 的概率；评估指标 ROC AUC。训练集 105,000 行、测试集 45,000 行，原始特征 56 列。

    本 notebook 读取 `data/` 中的原始 CSV，做字段审计、探索分析和数据质量检查，并生成后续所有 notebook 共用的**固定数据接口**（`artifacts/01_data_exploration/`）：统一类型的 56 列特征、标签、ID 和 seed=42 的固定五折。这里只做确定性的格式整理：不删除申请、不猜测异常值、不使用标签构造特征。

    {NOTEBOOK_MAP}

    默认不重写 `artifacts/01_data_exploration/`（已存在时只做逐项核对），保证之后所有实验结果目录的数据指纹不变。
    '''),
    code(SETUP),
    code('''
    import sklearn

    RANDOM_STATE = 42
    N_SPLITS = 5
    SCHEMA_VERSION = "1.0"
    MISSING_CATEGORY = "MISSING"
    REBUILD_PREPARED = False   # True：重写 artifacts/01_data_exploration（manifest 的哈希会改变，已有实验目录将无法续跑）
    pd.set_option("display.max_colwidth", 75)
    print(f"原始数据：{DATA_DIR}\\n准备数据：{PREPARED_DIR}\\nsklearn {sklearn.__version__}")
    '''),
    md('''
    ## 1. 读取原始文本并验证身份与标签

    全部先读成文本，避免 `00123` 丢失前导零，也避免 pandas 把合法类别当成缺失。检查字段顺序、ID、标签和提交模板；标签在转整数**之前**检查为精确的 `0`/`1`。
    '''),
    raw_code(cell('prep_read')),
    md('''
    ## 2. 字段类型与处理边界

    13 个数值字段保持原单位（贷款额、房价为美元，收入为千美元/年，LTV 为百分数）；其他 43 个字段按类别处理，包括数字类别代码、DTI 区间、年龄区间和地理代码。

    | 输入情况 | 统一处理 |
    |---|---|
    | 数值空白 | `NaN`；数值文本转换失败时直接报错 |
    | 类别空白 | 字符串 `MISSING` |
    | 整数类别代码 `1.0` | 规范为 `1`，不修改区间标签 |
    | `county_code` / `derived_msa-md` | 五位字符串；保留 `99999` 特殊类别 |
    | 不适用、无共同申请人、豁免（`1111`）等代码 | 保留各自原代码，与空白分开 |
    | 负收入、零值、极端 LTV、房龄 -1 | 保留并审计，不删除也不替换 |
    '''),
    raw_code(cell('prep_types')),
    md('## 3. 用同一个确定性函数整理 train / test\n\n函数不接收标签，也不在全数据上计算中位数、分位数或编码器。'),
    raw_code(cell('prep_normalize')),
    md('## 4. 验证信息和行顺序没有丢失'),
    raw_code(cell('prep_verify')),
    md('''
    ## 5. 数据概览

    下列分位数只用于描述，不作为模型阈值。对测试集只检查输入分布与接口。
    '''),
    raw_code(cell('prep_overview')),
    md('''
    ## 6. 探索分析：哪些信息与批准有关

    整体批准率约 85%。下图按关键字段列出批准率（虚线为整体批准率），可以看到：没走自动审批系统（`aus-1 = 6`）、高 DTI、LTV 缺失或过高的申请批准率明显更低。
    '''),
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
        ax.set_ylabel('approval rate')


    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    rate_bar(axes[0, 0], X_train['loan_type'], 'loan_type (1=conventional, 2=FHA, 3=VA, 4=RHS)')
    rate_bar(axes[0, 1], X_train['aus-1'], 'aus-1 (automated underwriting system; 6 = not used)')
    rate_bar(axes[0, 2], X_train['debt_to_income_ratio'], 'debt_to_income_ratio', order=dti_order)
    rate_bar(axes[1, 0], ltv_band, 'loan_to_value_ratio band', order=ltv_order)
    rate_bar(axes[1, 1], X_train['applicant_credit_score_type'], 'applicant_credit_score_type')
    rate_bar(axes[1, 2], X_train['occupancy_type'], 'occupancy_type (1=principal residence)')
    plt.tight_layout()
    plt.show()
    print(f'整体批准率：{overall_rate:.3f}')
    '''),
    md('**缺失本身就是信号**：关键字段缺失的申请批准率明显不同。树模型把缺失当成单独一类（`nan_mode=Min`），所以这里不填补、不删除。'),
    code('''
    missing_signal = []
    for column in ['loan_to_value_ratio', 'property_value', 'income', 'loan_term']:
        mask = X_train[column].isna()
        missing_signal.append({'字段': column, '缺失行数': int(mask.sum()),
                               '缺失时批准率': y[mask].mean(), '有值时批准率': y[~mask].mean()})
    for column in ['debt_to_income_ratio', 'applicant_age']:
        mask = X_train[column].eq(MISSING_CATEGORY)
        missing_signal.append({'字段': column, '缺失行数': int(mask.sum()),
                               '缺失时批准率': y[mask].mean(), '有值时批准率': y[~mask].mean()})
    display(pd.DataFrame(missing_signal).round(3))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (column, use_log, clip) in zip(axes, [('loan_amount', True, None), ('income', True, None),
                                                    ('property_value', True, None), ('loan_to_value_ratio', False, (0, 150))]):
        for label, colour, name in [(1, '#4c72b0', 'approved'), (0, '#dd8452', 'rejected')]:
            values = X_train.loc[y.eq(label) & X_train[column].notna(), column]
            if use_log:
                values = np.log10(values[values > 0])
            if clip:
                values = values.clip(*clip)
            ax.hist(values, bins=60, density=True, alpha=0.5, color=colour, label=name)
        ax.set_title(column + (' (log10)' if use_log else ', clipped 0-150'))
    axes[0].legend()
    plt.tight_layout()
    plt.show()
    '''),
    md('''
    ## 7. 风险记录与审计计数

    负收入、零收入、大额申请和 LTV 高于 100% 都保留；重复特征行先报告，不删除。
    '''),
    raw_code(cell('prep_audit')),
    md('''
    ## 8. 数据质量检查

    - **对抗验证**：训练一个模型区分训练行和测试行。AUC ≈ 0.5 说明两者分布一致，交叉验证的结果可以代表测试集。
    - **id 结构**：id 能否预测标签、相邻 id 的标签是否相关（检查按机构或时间排序造成的泄漏）。
    - **派生字段一致性**：派生字段与原字段是否一致（判断数据是真实记录还是生成数据）。
    - **「异常值」是不是错误**：极端或可疑取值所在组的批准率。
    '''),
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
        ('训练/测试对抗验证 AUC（0.5 = 分布一致）', f'{adversarial_auc:.4f}'),
        ('只用 id 预测标签的 AUC', f'{roc_auc_score(y, train_id_values):.4f}'),
        ('相邻 10 个 id 的平均标签与自身标签的相关', f'{neighbour_corr:.4f}'),
        ('id 范围：训练 / 测试', f'{train_id_values.min()}–{train_id_values.max()} / '
                               f'{test_id_values.min()}–{test_id_values.max()}'),
        ('产品类型 = 贷款类型 + 留置顺序', f"{(loan_type_name + ':' + lien_name).eq(X_train['derived_loan_product_type']).mean():.2%}"),
        ('住宅类别与施工方式一致',
         f"{X_train['derived_dwelling_category'].str.split(':').str[1].eq(construction_name).mean():.2%}"),
    ], columns=['检查', '结果'])
    display(checks)

    loan_to_value_raw = X_train['loan_amount'] / X_train['property_value']
    anomalies = {
        '全部': pd.Series(True, index=X_train.index),
        'LTV 缺失': ltv.isna(),
        'LTV > 150': ltv > 150,
        '普查区人口 = 0': X_train['tract_population'].eq(0),
        '地区收入 = 0': X_train['ffiec_msa_md_median_family_income'].eq(0),
        '贷款额 / 房价 > 2': loan_to_value_raw > 2,
        '收入 ≤ 0': X_train['income'] <= 0,
    }
    display(pd.DataFrame([{'情况': name, '行数': int(mask.sum()), '批准率': y[mask].mean()}
                          for name, mask in anomalies.items()]).round(3))
    '''),
    md('''
    **结论**：训练和测试分布一致、id 不含信息、派生字段与原字段一致（真实 HMDA 记录）。「异常值」所在组的批准率远低于整体，它们是**拒绝信号**而不是录入错误，因此全部保留、交给模型学习。
    '''),
    md('''
    ## 9. 固定交叉验证折

    `StratifiedKFold(5, shuffle=True, random_state=42)`，折号 1–5。之后所有实验都使用这同一套折，比较时只看同一折上的配对差。
    '''),
    raw_code(cell('prep_folds')),
    md('''
    ## 10. 输出固定数据接口

    `train_features.csv` / `test_features.csv` 只含 56 个特征；`train_labels.csv`、`test_ids.csv`、`folds.csv` 记录 ID、标签和折号；`schema.json` 描述类型，`manifest.json` 记录输入/输出的 SHA-256。后续 notebook 都会先核对这些哈希。已存在时不重写（见 `REBUILD_PREPARED`）。
    '''),
    raw_code(cell('prep_write')),
    md('## 11. 回读验证\n\n用本次重新整理的结果，逐项核对已保存的文件：数值、类别、列顺序、ID、折号和 SHA-256。'),
    raw_code(cell('prep_readback')),
]

# ============================================================================= 02
ROUND1_TABLE = '''
| 特征块 | 新增列 | 想法 |
|---|---|---|
| `dti` | `dti_lower`、`dti_upper` | DTI 类别（如 `30%-<36%`）对应的数值上下界 |
| `ltv` | `ltv_calc`、`ltv_gap` | 由贷款额 / 房价**重新算出** LTV，以及它与申报 LTV 的差 |
| `income` | `income_to_area`、`loan_to_income`、`value_to_income` | 收入相对地区收入、贷款额和房价相对收入 |
| `missing` | `missing_pattern`（类别）、`missing_count` | 关键字段的缺失组合 |
| `freq` | `freq_county_code`、`freq_derived_msa-md`、`freq_state_code` | 地区在训练 + 测试中出现的次数（不使用标签） |
'''

ROUND2_TABLE = '''
| 特征块 | 新增列 | 依据 |
|---|---|---|
| `precision` | `ltv_precision`（缺失 / 整数 / 1–3 位小数 / 更多） | LTV 申报为整数时批准率 90.7%，两位小数 84.4%：反映不同机构的申报习惯，树模型无法从数值本身切出「是否整数」 |
| `thresholds` | `ltv_band`（类别）、`ltv_over_program` | 80、95、97、100 是常见产品门槛；`ltv_over_program` = LTV − 该贷款类型通常上限（常规 97、FHA 96.5、VA/RHS 100） |
| `implied` | `implied_property_value`、`implied_value_ratio` | 由贷款额和申报 LTV 反推房价，再与申报房价比较 |
| `term` | `term_is_360`、`term_years` | 贷款期限是否为 30 年 |
'''

ROUND3_TABLE = '''
| 特征块 | 新增列 | 依据 |
|---|---|---|
| `purpose` | `product_flags`（7 个产品标志拼成一个类别） | 残差诊断中离散度最高的分组（仅 17 组，证据弱） |
| `ltv_consistency` | `ltv_consistency`、`ltv_outside_interval` | 贷款额与房价都是 1 万美元区间的中点，由它们只能确定 LTV 的区间；申报值落在区间外说明房价口径不同 |
| `risk_layers` | `risk_layer_count`、`risk_layer_pattern` | 审批中常见的「风险叠加」：高 LTV、高 DTI、未用 AUS、非自住、LTV 缺失、次级留置 |
| `tract_income` | `income_to_tract` | 收入相对所在普查区的中位收入 |
'''

STORE_SAVE = '''
# 保存特征库：全部特征块的新列（不含原始 56 列，原始列由 01 的固定数据接口提供）
FEATURE_SETS = {
    'ROUND1_ALL': ['dti', 'ltv', 'income', 'missing', 'freq'],                                   # 第一轮全部（68 列）
    'G7': ['dti', 'ltv', 'income', 'missing', 'precision', 'thresholds', 'implied', 'term'],      # 最终 M1/M2（72 列）
    'LEAN': ['dti', 'ltv', 'income', 'missing', 'precision', 'thresholds'],                      # 最终 M3（68 列）
}
block_columns = {name: list(train_block.columns) for name, (train_block, _) in BLOCKS.items()}
all_columns = [c for columns in block_columns.values() for c in columns]
require(len(all_columns) == len(set(all_columns)), '不同特征块之间有重名列')
train_store = pd.concat([BLOCKS[name][0] for name in block_columns], axis=1)
test_store = pd.concat([BLOCKS[name][1] for name in block_columns], axis=1)
require(train_store.shape == (len(X_base), len(all_columns))
        and test_store.shape == (len(X_test_base), len(all_columns)), '特征库形状不符')
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
display(pd.DataFrame([{'特征块': name, '列数': len(cols), '列': ', '.join(cols)} for name, cols in block_columns.items()]))
print(f'特征库已保存到 {FEATURE_DIR}：{len(all_columns)} 列，回读核对通过。')
'''

def ablation_results(directory, baseline, min_gain, title):
    return f'''
runs = load_runs(ARTIFACTS / '02_feature_engineering' / '{directory}')
table, delta = paired_table(runs, 'variant', '{baseline}', min_gain={min_gain})
table['feature_count'] = runs.groupby('variant')['feature_count'].first()
display(table.round(6))
plot_deltas(delta, '{baseline}', '{title}')
'''

nb02 = [
    md('''
    # 02 · 特征工程

    **原则**：只给模型加入它**无法通过阈值切分自己得到**的信息；全部是确定性变换，不使用标签，不删除也不修改原始 56 列。数据检查（01）表明数据没有错误需要清洗，所以本项目的数据处理都是「加信息」。

    共三轮：
    1. **第一轮**：DTI 数值化、LTV 重算、收入比值、缺失组合、地区频次；
    2. **第二轮**：沿第一轮最有效的 LTV 方向继续——LTV 申报精度、产品门槛、反推房价、期限；
    3. **第三轮**：由残差诊断（05）提出的候选——产品标志组合、LTV 区间一致性、风险叠加、相对普查区收入。

    每一轮都用**消融实验**评估：同一套快速 CatBoost 配置、固定五折，只比较同一折上相对基线的 AUC 差；五折平均提升达标且至少 4/5 折为正才保留。最后把全部特征块保存为**特征库**（`artifacts/02_feature_engineering/feature_store/`），供 03–05 使用。

    {NOTEBOOK_MAP}
    '''),
    code(SETUP),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'ablation_round1': False,   # 第一轮消融：7 个变体 × 5 折，约 70 分钟
        'ablation_round2': False,   # 第二轮消融：8 个变体 × 5 折，约 80 分钟
        'ablation_round3': False,   # 第三轮消融：6 个变体 × 5 折，约 60 分钟
    }
    '''),
    md('## 1. 读取并核对固定数据接口\n\n核对 01 生成的 `artifacts/01_data_exploration/`：SHA-256、列顺序、ID 和固定五折。'),
    raw_code(DATA),
    md('## 2. 第一轮特征\n\n### DTI 上下界\n\n保留原始 DTI 类别；开放区间和缺失的未知端为 `NaN`，遇到未见过的类别直接报错。'),
    raw_code(cell('dti_bounds')),
    md(f'### 第一轮特征块\n\n比值的分母必须大于 0，否则为缺失，不加任意 epsilon。\n\n{ROUND1_TABLE}'),
    raw_code(cell('features_round1')),
    md(f'## 3. 第二轮特征\n\n`ltv_precision` 需要申报 LTV 的**原始文本**（例如 `80.0` 与 `79.85`），所以按字符串重新读取这一列。\n\n{ROUND2_TABLE}'),
    raw_code(cell('features_round2')),
    md(f'## 4. 第三轮特征\n\n{ROUND3_TABLE}'),
    raw_code(cell('features_round3')),
    md('## 5. 保存特征库\n\n03–05 读取这里保存的特征，不再重复定义。文件带 SHA-256 与数据指纹，数据变化时会拒绝读取。'),
    raw_code(STORE_SAVE),
    md('''
    ## 6. 消融实验

    **快速配置**：CatBoost GPU，depth 8、学习率 0.05、L2=3、Bernoulli 抽样 0.8、固定 1600 轮。每个变体 = 原始 56 列 + 若干特征块；五折上与基线**同折配对**比较。下表 `delta_vs_base` 为五折平均的 AUC 变化，`folds_better` 为 5 折中优于基线的折数，`keep` 为是否达到保留标准。
    '''),
    raw_code(RESULT_HELPERS),
    md('### 6.1 第一轮（基线 = 原始 56 列）\n\n保留标准：平均 ≥ +0.0003 且 ≥ 4/5 折。'),
    raw_code(cell('ablation_r1_config')),
    gated("TRAIN['ablation_round1']", cell('ablation_run') + '\n\n' + cell('ablation_r1_summary')),
    code(ablation_results('ablation_round1', 'F0_base', 0.0003, 'Feature ablation, round 1')),
    md('''
    **结论**：全部特征块一起 **+0.0030（5/5 折）**；单独看，LTV 重算（`ltv_calc`、`ltv_gap`）**+0.0021（5/5）** 是主要来源，其余单独几乎无效，地区频次略差。
    '''),
    md('### 6.2 第二轮（基线 G0 = 第一轮全部特征）\n\n保留标准：平均 ≥ +0.0003 且 ≥ 4/5 折。'),
    raw_code(cell('ablation_r2_config')),
    gated("TRAIN['ablation_round2']", cell('ablation_run') + '\n\n' + cell('ablation_r2_summary')),
    code(ablation_results('ablation_round2', 'G0_all', 0.0003, 'Feature ablation, round 2')),
    md('''
    **结论**：**G7（去掉地区频次 + 全部第二轮特征，72 列）+0.00125（5/5）** 最好；LTV 申报精度 +0.00075（4/5）、LTV 产品门槛 +0.00074（5/5）单独有效；反推房价、期限单独不确定。
    '''),
    md('### 6.3 第三轮（基线 H0 = G7）\n\n不同 Colab 会话之间同一配置可相差约 0.0005，所以本轮保留标准提高为：平均 ≥ +0.0005 且 ≥ 4/5 折。'),
    raw_code(cell('ablation_r3_config')),
    gated("TRAIN['ablation_round3']", cell('ablation_run') + '\n\n' + cell('ablation_r3_summary')),
    code(ablation_results('ablation_round3', 'H0_G7', 0.0005, 'Feature ablation, round 3')),
    md('''
    **结论**：第三轮最好只有 +0.00035（3/5），均未达标，与残差诊断（05）一致——特征工程到此饱和。

    ## 7. 最终使用的特征集

    | 特征集 | 组成 | 列数 | 用于 |
    |---|---|---:|---|
    | **G7** | 原始 56 列 + `dti`、`ltv`、`income`、`missing`、`precision`、`thresholds`、`implied`、`term` | 72 | 最终模型 M1、M2 |
    | **LEAN** | 原始 56 列 + `dti`、`ltv`、`income`、`missing`、`precision`、`thresholds` | 68 | 最终模型 M3 |
    | ROUND1_ALL | 原始 56 列 + 第一轮全部 | 68 | 对照 M0 |

    第三轮特征保存在特征库中，但没有进入最终模型。
    '''),
]

# ============================================================================= 03
BASELINE_RESULTS = '''
cpu_run = ARTIFACTS / '03_model_selection' / 'baseline_catboost_xgboost'   # CatBoost R1/R2、XGBoost（CPU）
lr_run = ARTIFACTS / '03_model_selection' / 'baseline_logistic'            # 逻辑回归
rows = []
for path in [cpu_run / 'summary.csv', lr_run / 'summary.csv', cpu_run / 'blend_summary.csv']:
    for r in pd.read_csv(path).itertuples():
        rows.append({'模型': r.model, '五折 AUC': r.mean_fold_auc, '标准差': r.std_fold_auc,
                     'OOF LogLoss': r.pooled_oof_logloss})
rows.append({'模型': 'CatBoost_R2（GPU T4，同参数）', '五折 AUC': 0.91498, '标准差': np.nan, 'OOF LogLoss': np.nan})
display(pd.DataFrame(rows).sort_values('五折 AUC', ascending=False).reset_index(drop=True).round(5))
'''

BASELINE_RERUN = '''
# 重新训练 CatBoost R1 / R2（CPU，固定 1600 轮；与原基线协议相同），结果写入单独目录，不覆盖原结果
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
    for name, depth in [('CatBoost_R1', 6), ('CatBoost_R2', 8)]:
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
        print(f"{name} fold {fold}: AUC={row['valid_auc']:.6f}  {row['seconds'] / 60:.1f} 分钟", flush=True)
display(pd.DataFrame(rerun_rows).pivot(index='fold', columns='model', values='valid_auc').agg(['mean', 'std']).round(6))
'''

TRANSFORM_RESULTS = '''
TRANSFORM_DIR = ARTIFACTS / '03_model_selection' / 'transformations'
transform = pd.read_csv(TRANSFORM_DIR / 'auc_summary.csv')
order = ['base', 'poly2', 'poly3', 'quantile', 'rbf', 'spline']
display(transform.pivot(index='representation', columns='model', values='auc_mean').reindex(order).round(5))
if (TRANSFORM_DIR / 'auc_comparison.png').is_file():
    display(Image(filename=str(TRANSFORM_DIR / 'auc_comparison.png'), width=760))
'''

TUNING_RESULTS = '''
TUNING_DIR = ARTIFACTS / '03_model_selection' / 'catboost_tuning'
tuning_runs = load_runs(TUNING_DIR)
fold1 = tuning_runs[tuning_runs['fold'].eq(1)].set_index('candidate')
screen = pd.DataFrame({
    '第 1 折 AUC': fold1['valid_auc'],
    '相对 B0': fold1['valid_auc'] - fold1.loc['B0_R2_fixed1600', 'valid_auc'],
    '重训轮数': fold1['refit_rounds'],
}).sort_values('第 1 折 AUC', ascending=False)
print('第一轮：第 1 折筛选')
display(screen.round(6))
finalists = ['B0_R2_fixed1600', 'C7_d8_lr03_l2_10_sub80', 'C10_d8_lr03_l2_30_sub80', 'C11_d6_lr03_l2_10_sub80']
table, delta = paired_table(tuning_runs[tuning_runs['candidate'].isin(finalists)], 'candidate', 'B0_R2_fixed1600')
print('第二轮：前几名五折复核')
display(table.round(6))
plot_deltas(delta, 'B0_R2_fixed1600', 'CatBoost tuning, 5 folds (vs R2 on GPU)')
'''

SCREEN_RESULTS = '''
screen_runs = load_runs(ARTIFACTS / '03_model_selection' / 'training_modes')
table, delta = paired_table(screen_runs, 'variant', 'V0_base')
display(table.round(6))
plot_deltas(delta, 'V0_base', 'CatBoost training modes, 5 folds')
'''

nb03 = [
    md('''
    # 03 · 模型选择与调参

    回答两个问题：**用哪一类模型**，以及 **CatBoost 用什么参数和训练方式**。所有比较都使用 01 的固定五折，只看同一折上的配对差。

    1. 基线：逻辑回归、XGBoost、CatBoost（CPU / GPU）、AutoGluon
    2. 线性模型与随机森林 + 数值变换
    3. CatBoost 调参（内层早停、学习率、深度、L2、抽样）
    4. CatBoost 训练方式（有序提升、生长方式、抽样、分箱、类别组合）

    LightGBM 与神经网络要和最终的 CatBoost 比较融合效果，放在 05。

    {NOTEBOOK_MAP}
    '''),
    code(SETUP),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'baseline_cpu': False,   # CatBoost R1/R2 CPU 基线（Colab CPU 上约 3 小时）
        'tuning': False,         # CatBoost 调参五折复核（GPU，约 2.5 小时）
        'screen': False,         # 训练方式筛选（GPU，约 2 小时）
    }
    '''),
    md('## 0. 读取固定数据接口与特征库'),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS),
    md('''
    ## 1. 基线模型

    | 模型 | 设置 |
    |---|---|
    | 逻辑回归 | 地理简化（N2）+ DTI×贷款类型交互 + LTV 尾部标记，折内填补与标准化 |
    | XGBoost | 原生类别，深度 6，折内早停 |
    | CatBoost R1 / R2 | 原始 56 列，深度 6 / 8，学习率 0.05，L2=3，固定 1600 轮 |

    CatBoost 明显最好。**同参数下 GPU 版比 CPU 版低约 0.001**——第 3 节会看到原因主要是两者默认的行抽样方式不同。逻辑回归与 XGBoost 的完整训练代码见 Git 历史第一个提交中的 `02_hmda_models.ipynb`；CatBoost 基线可用下方开关重训。
    '''),
    code(BASELINE_RESULTS),
    gated("TRAIN['baseline_cpu']", BASELINE_RERUN, gpu=False),
    md('''
    **AutoGluon**（`extreme_quality`、T4 GPU、2,400 秒，只跑第 1 折）：0.8996。TabICL、TabDPT 因内存不足失败，实际只训练出一个 LightGBM；这不代表 AutoGluon 的上限，但在可用资源下没有竞争力。
    '''),
    md('''
    ## 2. 线性模型、随机森林与数值变换

    在 Ridge 与随机森林上比较 6 种数值表示（原值、二阶、三阶、分位数正态、RBF 近似、样条），类别统一 One-Hot。变换对 Ridge 最多 +0.003、对随机森林无益，两者都远低于 CatBoost（≈ 0.916），因此后续只用 CatBoost。原始训练代码见 Git 历史第一个提交中的 `07_hmda_transformations_comparison.ipynb`。
    '''),
    code(TRANSFORM_RESULTS),
    md('''
    ## 3. CatBoost 调参

    **协议**：外层训练部分内再留出 15% 早停，得到最佳轮数 *b*，再在整个外层训练部分按 `round(b / 0.85)` 轮重训；外层验证只用于评分。B0 = GPU 上的 R2（学习率 0.05、固定 1600 轮）。先在第 1 折上筛 12 组，再对前几名做五折复核。
    '''),
    raw_code(cell('tuning_config')),
    gated("TRAIN['tuning']", cell('tuning_run') + '\n\n' + cell('tuning_summary'),
          setup="X_dti = pd.concat([X_base, BLOCKS['dti'][0]], axis=1)   # C8 使用的 DTI 上下界"),
    code(TUNING_RESULTS),
    md('''
    **结论**：**C7（depth 8、lr 0.03、L2=10、Bernoulli 抽样 0.8）+0.00156** 与 **C11（depth 6、其余相同）+0.00142**，均 5/5 折。只加 Bernoulli 抽样一项就贡献约 +0.001——GPU 默认 Bayesian 抽样、CPU 默认 MVS 抽样，这正是 GPU 版原来低约 0.001 的原因。depth 10 更差。最终模型使用 C7 与 C11。
    '''),
    md('''
    ## 4. CatBoost 训练方式筛选

    在最终特征（LEAN，68 列）上，检验 GPU 默认设置之外的选项：有序提升（为小数据设计，超过 5 万行时 GPU 默认关闭）、NewtonL2 打分、Depthwise / Lossguide 生长、Poisson 抽样、抽样 0.66、254 个分箱、不做类别组合、删除重复的人口统计列。基线 V0 ≈ 04 的 M3，学习率 0.06 加速；每折内层早停后直接给外层验证打分。保留标准：平均 ≥ +0.0005 且 ≥ 4/5 折。
    '''),
    raw_code(cell('screen_config')),
    gated("TRAIN['screen']", cell('screen_preflight') + '\n\n' + cell('screen_run') + '\n\n' + cell('screen_summary')),
    code(SCREEN_RESULTS),
    md('**结论**：没有一项达标；254 个分箱 +0.00023（4/5）最好，其余持平或更差。现有设置（对称树、Plain 提升、Cosine 打分、Bernoulli 0.8）已是这些选项中最好的。'),
    md('''
    ## 5. 小结

    最终模型为 **CatBoost（GPU）**，参数 **C7 / C11**；其他训练方式都没有带来可靠提升。04 把这两组参数与 02 的特征集组合成最终模型；LightGBM 与神经网络能否与它互补，在 05 中检验。
    '''),
]

# ============================================================================= 04
CV_RESULTS = '''
cv_runs = load_runs(CV_DIR)
per_group = cv_runs.pivot(index='fold', columns='group', values='valid_auc')
display(per_group.agg(['mean', 'std']).T.sort_values('mean', ascending=False).round(6))

oof = {g: read_oof(CV_DIR, g) for g in ['M0_C11_all'] + [b + t for b in CAT_BASE_GROUPS for t in SEED_TAGS.values()]}
reference = oof['M0_C11_all']
for name, frame in oof.items():
    require(frame['id'].equals(reference['id']) and frame['fold'].equals(reference['fold']), f'{name} 的 OOF 未对齐')
blend_seed42 = np.mean([oof[b]['prob'] for b in CAT_BASE_GROUPS], axis=0)
b_seeds = np.mean([np.mean([oof[b + t]['prob'] for t in SEED_TAGS.values()], axis=0) for b in CAT_BASE_GROUPS], axis=0)
candidates = {
    'M0: C11 + round-1 features': reference['prob'],
    'M3: C11 + LEAN': oof['M3_C11_lean']['prob'],
    'M1+M2+M3 (seed 42)': blend_seed42,
    'B_seeds: M1/M2/M3 x 3 seeds': b_seeds,
}
base_folds = fold_auc(reference, reference['prob'])
fold_table = pd.DataFrame({name: fold_auc(reference, prob) for name, prob in candidates.items()})
summary = pd.DataFrame({'五折 AUC': fold_table.mean(), '相对 M0': fold_table.sub(base_folds, axis=0).mean(),
                        '更好折数': fold_table.sub(base_folds, axis=0).gt(0).sum()})
display(summary.round(6))

fig, ax = plt.subplots(figsize=(7, 3.8))
for name in fold_table.columns:
    ax.plot(fold_table.index, fold_table[name], marker='o', label=name)
ax.set_xticks(range(1, 6))
ax.set_xlabel('fold')
ax.set_ylabel('validation AUC')
ax.set_title('Final 5-fold models')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
'''

LC_RESULTS = '''
lc = load_runs(FULL_DIR, 'LC_*__fold?.json')
curve = lc.pivot(index='fold', columns='name', values='valid_auc')
curve['gain_84k_vs_67k'] = curve['LC_full'] - curve['LC_sub80']
display(curve.round(6))
print(f"训练数据 ×1.25（67,200 → 84,000 行）：平均 AUC {curve['gain_84k_vs_67k'].mean():+.5f}，"
      f"{int((curve['gain_84k_vs_67k'] > 0).sum())}/5 折为正")

fig, ax = plt.subplots(figsize=(6, 3.8))
for fold, row in curve.iterrows():
    ax.plot([67200, 84000], [row['LC_sub80'], row['LC_full']], marker='o', label=f'fold {fold}')
ax.set_xlabel('training rows')
ax.set_ylabel('validation AUC')
ax.set_title('Learning curve: M3, 80% vs 100% of outer-train')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
'''

FULL_RESULTS = '''
full_runs = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in sorted(FULL_DIR.glob('FULL_*.json'))])
display(full_runs[['name', 'base', 'seed', 'train_rows', 'rounds']].sort_values(['base', 'seed']))
full_pred = np.mean([mean_test([FULL_DIR / f'FULL_{b}_seed{s}__test.csv' for s in SEED_TAGS]) for b in CAT_BASE_GROUPS], axis=0)
fold_pred = np.mean([mean_test([CV_DIR / f'{b}{t}__fold{k}__test.csv' for t in SEED_TAGS.values() for k in range(1, 6)])
                     for b in CAT_BASE_GROUPS], axis=0)
checks = []
for label, path, expected in [
    ('最终提交 full_Bseeds', FULL_DIR / 'submission_full_Bseeds.csv', full_pred),
    ('备选 hybrid_50_50', FULL_DIR / 'submission_hybrid_50_50.csv', 0.5 * full_pred + 0.5 * fold_pred),
]:
    saved = pd.read_csv(path, dtype={'id': str})
    require(saved['id'].equals(sample_submission['id']), f'{path.name} 的 ID 顺序与提交模板不同')
    checks.append({'文件': label, '与重新计算的最大差异': float(np.abs(saved['accepted'].to_numpy() - expected).max())})
display(pd.DataFrame(checks))
print(f'全量模型与折平均模型的测试预测相关系数：{np.corrcoef(full_pred, fold_pred)[0, 1]:.5f}')
'''

PUBLIC_TABLE = '''
public = pd.DataFrame([
    ('CPU CatBoost R2, 56 raw features', 0.91604, 0.91857),
    ('GPU CatBoost R2 (same parameters)', 0.91498, 0.91764),
    ('C11 + round-1 features (M0)', 0.91931, 0.92073),
    ('M1+M2+M3 (seed 42), fold average', 0.92051, 0.92190),
    ('B_seeds: M1/M2/M3 x 3 seeds, fold average', 0.92073, 0.92197),
    ('hybrid: 50% full-data + 50% fold average', np.nan, 0.92259),
    ('full_Bseeds: 9 models on all 105k rows (FINAL)', np.nan, 0.92301),
], columns=['submission', '5-fold AUC', 'public LB AUC'])
display(public)
'''

nb04 = [
    md('''
    # 04 · 最终模型与提交

    最终模型 = **CatBoost（GPU）三组配置 × 3 个随机种子，全部在 105,000 行上训练**，测试概率先按组平均种子、再三组等权平均。公开榜 **0.92301**。

    | 组 | 参数（03） | 特征（02） |
    |---|---|---|
    | M1 | C11：depth 6、lr 0.03、L2=10、Bernoulli 0.8 | G7（72 列） |
    | M2 | C7：depth 8、lr 0.03、L2=10、Bernoulli 0.8 | G7 |
    | M3 | C11 | LEAN（68 列） |
    | M0（对照） | C11 | 第一轮全部特征（68 列） |

    步骤：① 五折验证每一组与它们的融合；② 学习曲线——判断「更多训练数据」是否有价值；③ 在全部训练数据上重训并生成提交文件。

    {NOTEBOOK_MAP}
    '''),
    code(SETUP),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'cv_models': False,        # 五折模型：M0 与 M1–M3 × 3 个种子（GPU，约 6 小时）
        'learning_curve': False,   # 学习曲线：10 次训练（GPU，约 75 分钟）
        'full_refit': False,       # 全量重训：9 个模型（GPU，约 50 分钟）
    }
    '''),
    md('## 0. 读取固定数据接口与特征库'),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS + '\n\n' + OOF_HELPERS),
    md('''
    ## 1. 五折模型与融合

    **协议**：每个外层折的训练部分内留出 15% 早停，得到最佳轮数 *b*，再在整个外层训练部分按 `round(b / 0.85)` 轮重训；外层验证只评分。每个折模型同时预测测试集。M1/M2/M3 各训练 3 个种子（42、2027、31415）。
    '''),
    raw_code(cell('cv_config')),
    gated("TRAIN['cv_models']", cell('cv_run') + '\n\n' + cell('cv_blend')),
    code(CV_RESULTS),
    md('''
    **结论**：第二轮特征让单组提升 +0.0006～+0.0009；三组融合 0.92051；每组 3 个种子再平均 **0.92073**（相对 M0 +0.0014，5/5 折）。这是能直接用五折验证的最好结果。
    '''),
    md('''
    ## 2. 学习曲线：更多训练数据值多少？

    上面的提交方式是「五个折模型的测试预测平均」，每个模型只见过 80% 的训练数据（84,000 行）。在同一折上，M3 分别用外层训练部分的 80%（67,200 行）和 100%（84,000 行）训练，比较验证 AUC。67k→84k 与 84k→105k 都是 ×1.25，前者的收益就是全量重训收益的估计。
    '''),
    raw_code(cell('refit_config')),
    gated("TRAIN['learning_curve'] or TRAIN['full_refit']", cell('refit_functions')),
    gated("TRAIN['learning_curve']", cell('refit_learning_curve')),
    code(LC_RESULTS),
    md('**结论**：训练数据 ×1.25 → AUC **+0.0012，5/5 折为正**。模型受数据量限制，用上全部 105,000 行是值得的。'),
    md('''
    ## 3. 全量重训与提交文件

    M1/M2/M3 × 3 个种子在全部 105,000 行上训练，轮数 = 该组五折重训轮数均值 × 1.25（与 `b / 0.85` 同一规则）。全量模型没有留出的验证集，是否采用只依据第 2 节的学习曲线。下面重新计算两份提交文件，并与保存的文件核对。
    '''),
    gated("TRAIN['full_refit']", cell('refit_full') + '\n\n' + cell('refit_submit')),
    code(FULL_RESULTS),
    md('''
    ## 4. 提交与公开榜

    公开榜只用约 30% 的测试数据（≈ 13,500 行），相差不到约 0.001 的提交，公开榜顺序不可靠（见 05），所以选模以五折和学习曲线为准；下面的公开榜分数顺序与它们的预测一致。
    '''),
    code(PUBLIC_TABLE),
    md('''
    **最终提交**：`submission_full_Bseeds.csv`（主选，公开榜 0.92301）与 `submission_hybrid_50_50.csv`（备选，0.92259）；副本在项目根目录的 `submissions/`。
    '''),
]

# ============================================================================= 05
RESIDUAL = '''
m3 = read_oof(CV_DIR, 'M3_C11_lean')
diag = X_base.assign(id=train_ids.astype('int64').to_numpy(), accepted=y.to_numpy()).merge(
    m3[['id', 'prob']], on='id', validate='one_to_one')
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
    ('普查区', diag[tract_columns].astype(str).agg('|'.join, axis=1), 5),
    ('县', diag['county_code'], 5),
    ('MSA', diag['derived_msa-md'], 5),
    ('州', diag['state_code'], 5),
    ('机构画像：AUS × 提交方式 × 支付对象 × 信用评分类型 × 预批',
     diag[['aus-1', 'submission_of_application', 'initially_payable_to_institution',
           'applicant_credit_score_type', 'preapproval']].astype(str).agg('|'.join, axis=1), 5),
    ('随机分组（对照，约 8000 组）', rng.integers(0, 8000, len(diag)), 5),
    ('贷款类型 × DTI', diag['loan_type'] + '|' + diag['debt_to_income_ratio'], 30),
    ('LTV 区间 × DTI', pd.cut(ltv, [0, 60, 80, 90, 95, 97, 100, np.inf]).astype(str) + '|' + diag['debt_to_income_ratio'], 30),
    ('申请人 × 共同申请人年龄', diag['applicant_age'] + '|' + diag['co-applicant_age'], 30),
    ('产品标志组合', diag[product_flags].astype(str).agg('|'.join, axis=1), 30),
    ('随机分组（对照，300 组）', rng.integers(0, 300, len(diag)), 30),
]
display(pd.DataFrame([{'分组': name, '标准化残差平方均值': dispersion(groups, min_rows)[0],
                       '组数': dispersion(groups, min_rows)[1], '最少行数': min_rows}
                      for name, groups, min_rows in groupings]).round(3))
'''

ANOMALY = '''
loan_value = diag['loan_amount'] / diag['property_value']
cases = {
    'LTV 缺失': ltv.isna(), 'LTV > 150': ltv > 150,
    '普查区人口 = 0': diag['tract_population'].eq(0), '地区收入 = 0': diag['ffiec_msa_md_median_family_income'].eq(0),
    '贷款额 / 房价 > 2': loan_value > 2, '收入 ≤ 0': diag['income'] <= 0,
}
rows = []
for name, mask in cases.items():
    rows.append({'情况': name, '行数': int(mask.sum()), '实际批准率': diag.loc[mask, 'accepted'].mean(),
                 '模型平均预测': diag.loc[mask, 'prob'].mean(),
                 '残差 z': residual[mask].sum() / np.sqrt(variance[mask].sum())})
display(pd.DataFrame(rows).round(3))
for threshold in [0.95, 0.98]:
    wrong = ((diag['prob'] > threshold) & diag['accepted'].eq(0)) | ((diag['prob'] < 1 - threshold) & diag['accepted'].eq(1))
    print(f'把握 ≥ {threshold:.0%} 却判错：{int(wrong.sum())} 行（{wrong.mean():.2%}）')

key_columns = ['county_code', 'tract_population', 'tract_minority_population_percent',
               'tract_to_msa_income_percentage', 'tract_owner_occupied_units', 'loan_amount', 'income', 'applicant_age']
both = pd.concat([X_base[key_columns], X_test_base[key_columns]], ignore_index=True).astype(str).agg('|'.join, axis=1)
group_size = both.map(both.value_counts()).iloc[:len(X_base)].to_numpy()
print(f'与其他申请「普查区 + 金额 + 收入 + 年龄」完全相同的训练行：{(group_size >= 2).mean():.2%}')
'''

BLEND_SEARCH = '''
from scipy.special import logit

reference = read_oof(CV_DIR, 'M0_C11_all')
pool = {}
for base in CAT_BASE_GROUPS:
    pool[f'{base} x3 seeds'] = np.mean([read_oof(CV_DIR, base + t)['prob'] for t in SEED_TAGS.values()], axis=0)
pool['B_seeds'] = np.mean([pool[f'{b} x3 seeds'] for b in CAT_BASE_GROUPS], axis=0)
pool['M0'] = reference['prob'].to_numpy()
pool['LightGBM'] = np.mean([read_oof(LGBM_DIR, g)['prob'] for g in ['L1_lgbm_leaves31', 'L2_lgbm_leaves127']], axis=0)
cpu = pd.read_csv(ARTIFACTS / '03_model_selection' / 'baseline_catboost_xgboost' / 'oof_predictions.csv')
cpu = cpu.sort_values('id', kind='stable').reset_index(drop=True)
require(cpu['id'].equals(reference['id']) and cpu['fold'].equals(reference['fold']), 'CPU 基线 OOF 未对齐')
for column in ['CatBoost_R1', 'CatBoost_R2', 'XGBoost']:
    pool[f'CPU {column}'] = cpu[f'prob_{column}'].to_numpy()
names = list(pool)
labels = reference['accepted'].to_numpy()
folds = reference['fold'].to_numpy()
logits = np.column_stack([logit(np.clip(pool[n], 1e-6, 1 - 1e-6)) for n in names])
display(pd.DataFrame({'五折 AUC': [fold_auc(reference, pool[n]).mean() for n in names],
                      '与 B_seeds 的相关': [np.corrcoef(pool[n], pool['B_seeds'])[0, 1] for n in names]},
                     index=names).sort_values('五折 AUC', ascending=False).round(5))


def mean_fold_auc(score, use_folds):
    return np.mean([roc_auc_score(labels[folds == k], score[folds == k]) for k in use_folds])


def greedy(use_folds, steps=15):
    start = names.index('B_seeds')
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
    rows.append({'留出的折': held_out,
                 '融合 AUC': roc_auc_score(labels[mask], blend[mask]),
                 'B_seeds AUC': roc_auc_score(labels[mask], pool['B_seeds'][mask]),
                 '选中的模型': ', '.join(f'{names[c]}×{chosen.count(c)}' for c in sorted(set(chosen)))})
nested = pd.DataFrame(rows)
nested['差'] = nested['融合 AUC'] - nested['B_seeds AUC']
display(nested.round(5))
print(f"嵌套交叉验证：平均 {nested['差'].mean():+.5f}，{int((nested['差'] > 0).sum())}/5 折更好")
'''

PSEUDO_RESULTS = '''
PL_DIR = ARTIFACTS / '05_error_analysis' / 'pseudo_label'
pl_runs = load_runs(PL_DIR, 'A_*__fold?.json')
pl_table = pl_runs.pivot(index='fold', columns='name', values='valid_auc')
pl_table['gain'] = pl_table['A_pseudo'] - pl_table['A_base']
display(pl_table.round(6))
print(f"软伪标签：平均 {pl_table['gain'].mean():+.5f}，{int((pl_table['gain'] > 0).sum())}/5 折为正")
'''

NOISE = '''
labels = reference['accepted'].to_numpy()
new, old = pool['B_seeds'], pool['M0']
rng = np.random.default_rng(SEED)
subset_auc, paired = [], []
for _ in range(300):
    idx = rng.choice(len(labels), 13500, replace=False)
    auc_new = roc_auc_score(labels[idx], new[idx])
    subset_auc.append(auc_new)
    paired.append(auc_new - roc_auc_score(labels[idx], old[idx]))
true_gap = roc_auc_score(labels, new) - roc_auc_score(labels, old)
print(f'全部 105,000 行上 B_seeds 比 M0 高 {true_gap:+.5f}')
print(f'随机 13,500 行（≈ 公开榜）：AUC 标准差 {np.std(subset_auc):.4f}；配对差标准差 {np.std(paired):.4f}；'
      f'公开榜显示顺序相反的概率 {np.mean(np.array(paired) < 0):.0%}')
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(np.array(paired) * 1e4, bins=30, color='#4c72b0')
ax.axvline(true_gap * 1e4, color='black', ls='--', label='difference on all rows')
ax.axvline(0, color='#c44e52', lw=1)
ax.set_xlabel('AUC(B_seeds) - AUC(M0) on 13.5k random rows (x1e-4)')
ax.set_title('How noisy is a 30% public leaderboard?')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

lc = load_runs(FULL_DIR, 'LC_full__fold?.json').set_index('fold')['valid_auc']
m3_runs = load_runs(CV_DIR, 'M3_C11_lean__fold?.json').set_index('fold')['valid_auc']
display(pd.DataFrame({'学习曲线中的 LC_full': lc, '五折模型中的 M3': m3_runs, '差': lc - m3_runs}).round(6))
'''

LGBM_RESULTS = '''
LGBM_DIR = ARTIFACTS / '05_error_analysis' / 'lightgbm'
lgbm_runs = load_runs(LGBM_DIR, 'L*_lgbm_*__fold?.json')
display(lgbm_runs.pivot(index='fold', columns='group', values='valid_auc').agg(['mean', 'std']).round(5))

cv_reference = read_oof(CV_DIR, 'M0_C11_all')
b_seeds = np.mean([np.mean([read_oof(CV_DIR, b + t)['prob'] for t in SEED_TAGS.values()], axis=0)
                   for b in CAT_BASE_GROUPS], axis=0)
lgbm_oof = [read_oof(LGBM_DIR, g) for g in sorted(lgbm_runs['group'].unique())]
for frame in lgbm_oof:
    require(frame['id'].equals(cv_reference['id']) and frame['fold'].equals(cv_reference['fold']),
            'LightGBM 的 OOF 与 CatBoost 未对齐')
lgbm = np.mean([frame['prob'] for frame in lgbm_oof], axis=0)
base_folds = fold_auc(cv_reference, b_seeds)
blend_rows = []
for weight in [0.0, 0.1, 0.2, 0.3, 1.0]:                  # 只看预设权重，不做搜索
    folds = fold_auc(cv_reference, (1 - weight) * b_seeds + weight * lgbm)
    blend_rows.append({'LightGBM 权重': weight, '五折 AUC': folds.mean(),
                       '相对 B_seeds': (folds - base_folds).mean(),
                       '更好折数': int((folds - base_folds).gt(0).sum())})
display(pd.DataFrame(blend_rows).round(6))
print(f'B_seeds 与 LightGBM 的 OOF 相关系数：{np.corrcoef(b_seeds, lgbm)[0, 1]:.4f}')
'''

TABM_RESULTS = '''
NN_DIR = ARTIFACTS / '05_error_analysis' / 'tabm'
tabm_runs = load_runs(NN_DIR, 'TabMMini__fold?.json')
display(tabm_runs[['fold', 'valid_auc', 'best_epoch', 'inner_stop_auc']].round(6))
display(pd.read_csv(NN_DIR / 'blend_by_fold.csv').round(6))
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
    md('''
    # 05 · 误差分析与未采用的方法

    最终模型之后，检查还有没有可挖的信息，并记录试过但没有采用的方法：

    1. **残差诊断**：模型在哪些分组上系统性地高估或低估？
    2. **其他模型族**：LightGBM 与神经网络 TabM-mini 能否与 CatBoost 互补？
    3. **跨实验融合**：把所有保存过的模型放在一起，用嵌套交叉验证挑选融合，能否超过最终方案？
    4. **测试集软伪标签**：学习曲线显示模型受数据量限制，用 45,000 个无标签测试行能否补充？
    5. **公开榜噪声与运行间差异**：为什么选模只看五折，以及决策阈值从何而来。

    {NOTEBOOK_MAP}
    '''),
    code(SETUP),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'lightgbm': False,       # LightGBM 五折（CPU 即可，约 30 分钟）
        'tabm': False,           # TabM-mini 神经网络（GPU，第 1 折约 5 分钟）
        'pseudo_label': False,   # 软伪标签逐折验证（GPU，约 2 小时；验证通过才做全量部分）
    }
    '''),
    md('## 0. 读取固定数据接口与特征库'),
    raw_code(DATA),
    raw_code(FEATURE_LOADER),
    raw_code(RESULT_HELPERS + '\n\n' + OOF_HELPERS),
    md('''
    ## 1. 残差诊断

    用 M3 的五折验证预测，按分组计算**标准化残差平方的均值** Σ(y − p)² / Σp(1 − p) 在各组上的平均：≈ 1 表示该分组上模型已经校准、没有遗漏的信息；明显大于随机分组说明模型缺了这部分信息。
    '''),
    code(RESIDUAL),
    md('''
    **结论**：地理（普查区、县、MSA）、「机构画像」、贷款类型 × DTI、LTV × DTI 都与随机分组相当，模型在这些方向上不缺信息。只有产品标志组合偏高（1.77，但只有 17 组，证据弱），它成了第三轮特征（02）的候选之一，消融后同样无效。
    '''),
    md('### 「异常值」与标签噪声\n\n模型在这些组上的平均预测与实际批准率一致——异常值是信号，清理会丢失信息；模型很有把握却判错的行很少，标签噪声不是问题。'),
    code(ANOMALY),
    md('''
    ## 2. 其他模型族

    ### 2.1 LightGBM

    G7 特征（与 M1/M2 相同），两个容量（31 / 127 个叶子），协议与 CatBoost 相同：内层早停 → 外层重训。先看单独的五折 AUC，再与最终 CatBoost（B_seeds）按预设权重融合：权重 0 = 只用 CatBoost，1 = 只用 LightGBM。
    '''),
    raw_code(cell('lgbm_config')),
    gated("TRAIN['lightgbm']", cell('lgbm_run'), gpu=False, catboost=False,
          setup='OUTPUT_DIR.mkdir(parents=True, exist_ok=True)'),
    code(LGBM_RESULTS),
    md('**结论**：LightGBM 单独约 0.911，比 CatBoost 低 0.009；与 CatBoost 的预测高度相关，加 10% 持平、更多权重变差，不采用。'),
    md('''
    ### 2.2 神经网络 TabM-mini

    32 个成员共享同一个 3 层 MLP，每个成员只在输入端有自己的 ±1 缩放向量（Gorishniy 等 2024 的简化版，PyTorch 实现）。数值列分位数正态化 + 16 段分段线性编码 + 缺失标记，类别列 8 维嵌入，全部只在外层训练部分上拟合。先跑第 1 折：单独 AUC 不到约 0.915 或融合无提升就停止。
    '''),
    raw_code(cell('tabm_config')),
    gated("TRAIN['tabm']", cell('tabm_prep') + '\n\n' + cell('tabm_run') + '\n\n' + cell('tabm_blend'),
          catboost=False, setup=TABM_SETUP),
    code(TABM_RESULTS),
    md('**结论**：TabM-mini 第 1 折 0.9088，比 CatBoost 低 0.006；与 CatBoost 的预测相关 0.984，按 10%–30% 融合都不提升，未跑完五折。'),
    md('''
    ## 3. 跨实验融合

    候选：最终方案 B_seeds、三组各自的 3 种子平均、M0、LightGBM、CPU 基线（CatBoost R1/R2、XGBoost）。在 4 折上用贪心法（logit 平均、可重复选择）挑选融合，在第 5 折上打分，轮换 5 次——得到的是诚实的估计。
    '''),
    code(BLEND_SEARCH),
    md('**结论**：旧模型与 B_seeds 高度相关且更弱，嵌套选择几乎总是只选 B_seeds 本身，融合没有可靠提升。'),
    md('''
    ## 4. 测试集软伪标签

    对每个外层折，老师 = 只在该折训练部分上训练的 9 个模型对测试集的平均预测；学生 = M3 + 45,000 个软标签测试行（CatBoost `CrossEntropy`，早停集只用真实行）。与不加伪标签的基线在同一次运行、同一折上比较；第 k 折验证标签始终不参与。
    '''),
    raw_code(cell('pseudo_config')),
    gated("TRAIN['pseudo_label']", cell('pseudo_functions') + '\n\n' + cell('pseudo_validate') + '\n\n' + cell('pseudo_final')),
    code(PSEUDO_RESULTS),
    md('**结论**：+0.00008（3/5），无效。对比 04 的学习曲线：多 25% **真实标签** +0.0012，多 54% **伪标签** ≈ 0——瓶颈是标签信息本身，而真实标签已经全部用上。'),
    md('''
    ## 5. 公开榜噪声与运行间差异

    用五折验证预测模拟公开榜：随机抽 13,500 行（≈ 测试集的 30%）计算 AUC。下面同时比较同一配置在两次不同 Colab 会话中的逐折结果，说明为什么小于约 0.0005 的提升被视为不确定。
    '''),
    code(NOISE),
    md('''
    **结论**：公开榜单个分数的抽样标准差约 0.004，不同提交的绝对分数只能粗略比较。两个高度相关的模型在同一批行上的**配对差**要稳定得多（B_seeds 对 M0 的标准差约 0.0006）：两者相差 0.0014 时，公开榜给出相反顺序的概率只有约 1%；但相差 0.0005 时约有两成概率颠倒。同一配置在两次 Colab 会话中逐折相差 0 到 0.0004，所以后期的保留标准提高到 +0.0005。
    '''),
    md('''
    ## 6. 结论

    - 数据干净，没有可清洗的错误；有效的数据处理都是「加信息」，且集中在 LTV 上。
    - 模型在各个分组上已经校准，换模型族、换训练方式、跨实验融合、伪标签都没有可靠收益。
    - 学习曲线显示模型**受标注数据量限制**：最后一步有效的提升来自用上全部训练数据（04）。
    - 公开榜只用约 30% 的测试行，相差不到约 0.001 的提交顺序不可靠；选模必须依据五折与学习曲线。

    不使用公开的 HMDA 原始数据：测试集本身来自该数据，用它训练会带入测试集的真实标签。
    '''),
]

save(nb01, '01_data_exploration.ipynb')
save(nb02, '02_feature_engineering.ipynb')
save(nb03, '03_model_selection.ipynb')
save(nb04, '04_final_model.ipynb')
save(nb05, '05_error_analysis.ipynb')
