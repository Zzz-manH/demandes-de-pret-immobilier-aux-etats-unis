# HMDA：美国住房贷款申请审批预测

预测每份申请被批准（`accepted = 1`）的概率，评估指标 ROC AUC。训练集 105,000 行、测试集 45,000 行，原始特征 56 列（13 数值 + 43 类别）。

**最终提交**：[submissions/20261016_FINAL_full_Bseeds.csv](submissions/20261016_FINAL_full_Bseeds.csv)，公开榜 **0.92301**（起点 0.91857）。全部实验数字见 [RESULTS.md](RESULTS.md)，提交记录见 [submissions/README.md](submissions/README.md)。

## 5 本 notebook

|notebook|内容|结果目录|原实验|
|---|---|---|---|
|[01_data_exploration](01_data_exploration.ipynb)|审计原始 CSV、探索分析、数据质量检查，生成固定的 56 列特征、标签、ID、seed=42 五折和哈希清单|`artifacts/01_data_exploration/`|01|
|[02_feature_engineering](02_feature_engineering.ipynb)|全部特征的定义（DTI 上下界、LTV 重算、LTV 申报精度、LTV 产品门槛等）、三轮特征消融、保存特征库|`artifacts/02_feature_engineering/`|10、12、14|
|[03_model_selection](03_model_selection.ipynb)|基线（逻辑回归、XGBoost、CatBoost、AutoGluon）、线性模型的数值变换、CatBoost 调参与训练方式筛选|`artifacts/03_model_selection/`|02、07、09、19|
|[04_final_model](04_final_model.ipynb)|五折模型 M0–M3 与多种子融合、学习曲线、全部数据重训与最终提交|`artifacts/04_final_model/`|13、15、17|
|[05_error_analysis](05_error_analysis.ipynb)|残差诊断与「异常值」、LightGBM 与 TabM-mini、跨实验融合、软伪标签、公开榜噪声与运行间差异|`artifacts/05_error_analysis/`|15（LightGBM）、16、18|

「实验 02、07、09–19」是实验的原始编号。RESULTS.md 和提交记录都沿用这些编号，对照见文末。

## 最终方案

CatBoost（GPU）三组配置 × 3 个随机种子（42、2027、31415），共 9 个模型，每个都在全部 105,000 行上训练（轮数 = 五折重训轮数均值 × 1.25）；测试概率先按组平均种子，再三组等权平均。

|组|参数|特征|
|---|---|---|
|M1|C11：depth 6、lr 0.03、L2=10、Bernoulli 0.8|G7：原始 56 列 + DTI 上下界、LTV 重算、收入比值、缺失组合、LTV 申报精度、LTV 产品门槛、反推房价、期限（72 列）|
|M2|C7：depth 8、lr 0.03、L2=10、Bernoulli 0.8|G7|
|M3|C11|原始 56 列 + DTI、LTV 重算、收入比值、缺失组合、LTV 精度、LTV 门槛（68 列）|

提升来自三处：围绕 LTV 的特征（约 +0.004）、调参（约 +0.0015）、用上全部训练数据（约 +0.001）。学习曲线显示模型受标注数据量限制（多 25% 数据 +0.0012，5/5 折）。

## 运行方式

- 整个项目文件夹放在 Google Drive 的 `Colab Notebooks/` 下，在 Colab 打开 notebook，「全部运行」。notebook 会自动找到项目里的 `data/`；本地 Jupyter 也可以运行，依赖见 `requirements.txt`。
- **按顺序运行 01 → 05**：01 生成固定数据接口（已存在时只逐项核对，不重写），02 生成特征库，03–05 读取两者。
- **默认只读**：所有训练开关为 `False`，只读取 `artifacts/` 中保存的结果，生成表格和图。CPU 运行时即可，每本几分钟。
- **重新训练**：
  - 把 notebook 顶部 `TRAIN` 中的一项改为 `True`，运行时类型选 **T4 GPU**（03 的 `baseline_cpu`、05 的 `lightgbm` 用 CPU 即可）。
  - 每个模型训练完立即写入 `artifacts/`，断线后重新运行会跳过已完成的部分。
  - 依赖在 notebook 内自动安装（CatBoost 固定 1.2.10）。
  - **复现最终提交**：04 中依次打开 `cv_models` 与 `full_refit`（全量重训的轮数取自五折模型）。
- 所有 notebook 都会先核对固定数据接口的哈希与五折，数据不一致时直接报错。
- `artifacts/` 约 280MB，不在 Git 中，完整副本在 Google Drive。
- 不使用公开的 HMDA 原始数据：测试集本身来自该数据，用它训练会带入测试集的真实标签。

## 目录

```
data/                                         原始数据（只读）：train.csv、test.csv、sample_submission.csv
01–05 *.ipynb                                 5 本 notebook
artifacts/                                    实验结果（不在 Git 中）
  01_data_exploration/                        固定特征、标签、ID、五折、哈希清单
  02_feature_engineering/ablation_round1–3    实验 10、12、14
  02_feature_engineering/feature_store/       02 运行时生成的特征库
  03_model_selection/baseline_logistic        实验 02：逻辑回归
  03_model_selection/baseline_catboost_xgboost 实验 02：CatBoost R1/R2、XGBoost
  03_model_selection/transformations          实验 07
  03_model_selection/catboost_tuning          实验 09
  03_model_selection/training_modes           实验 19
  04_final_model/cv_models                    实验 13、15：五折模型、OOF、测试预测
  04_final_model/full_refit                   实验 17：学习曲线、全量模型、最终提交
  05_error_analysis/lightgbm                  实验 15 中的 LightGBM
  05_error_analysis/tabm                      实验 16
  05_error_analysis/pseudo_label              实验 18
submissions/                                  最终提交文件与完整提交记录
tools/                                        生成 5 本 notebook 的脚本（见 tools/README.md）
RESULTS.md                                    全部实验结果
```

## 原实验编号对照

|原 notebook|内容|现在的位置|原结果目录 → 现在|
|---|---|---|---|
|01 `hmda_data_preparation`|审计原始 CSV，生成固定数据接口|01|`prepared` → `01_data_exploration`|
|02 `hmda_models`|基线：逻辑回归、XGBoost、CatBoost R1/R2（CPU）|03 第 1 节|`models/20260930T054023…`、`models/20260930T194248…` → `03_model_selection/baseline_logistic`、`baseline_catboost_xgboost`|
|07 `transformations_comparison`|Ridge / RandomForest + 数值变换|03 第 2 节|`transformation_models/20261005T153713…` → `03_model_selection/transformations`|
|09 `catboost_tuning`|CatBoost 调参，选出 C7 / C11|03 第 3 节|`gpu_tuning/catboost_screen_v1` → `03_model_selection/catboost_tuning`|
|10 `feature_ablation`|第一轮特征消融|02 第 6.1 节|`feature_ablation/feature_blocks_v1` → `02_feature_engineering/ablation_round1`|
|12 `feature_round2`|第二轮特征消融|02 第 6.2 节|`feature_ablation/feature_blocks_round2_v1` → `02_feature_engineering/ablation_round2`|
|13 `round2_final`|最佳参数 × 最佳特征的五折模型 M0–M3 与融合|04 第 1 节|`final_gpu/round2_final_v1` → `04_final_model/cv_models`|
|14 `feature_round3`|第三轮特征消融|02 第 6.3 节|`feature_ablation/feature_blocks_round3_v1` → `02_feature_engineering/ablation_round3`|
|15 `final_ensemble`|M1/M2/M3 加 2 个种子；LightGBM 融合检查|04 第 1 节、05 第 2.1 节|同 13；LightGBM 文件 → `05_error_analysis/lightgbm`|
|16 `tabm_blend`|神经网络 TabM-mini 与 CatBoost 融合（第 1 折）|05 第 2.2 节|`nn_models/tabm_mini_v1` → `05_error_analysis/tabm`|
|17 `full_refit`|学习曲线 + 全部数据重训 → 最终提交|04 第 2–3 节|`final_gpu/full_refit_v1` → `04_final_model/full_refit`|
|18 `pseudo_label`|测试集软伪标签|05 第 4 节|`final_gpu/pseudo_label_v1` → `05_error_analysis/pseudo_label`|
|19 `data_efficiency`|CatBoost 训练方式筛选|03 第 4 节|`gpu_tuning/data_efficiency_screen_v1` → `03_model_selection/training_modes`|

03、04、05、08、11 在整理前已被后续实验取代并删除（数字保留在 RESULTS.md 和提交记录中），06 从未存在。上表中的原始 notebook（带 Colab 运行输出）和生成它们的脚本保存在 Git 历史的第一个提交中。
