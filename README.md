# HMDA：美国住房贷款申请审批预测

预测每份申请被批准（`accepted = 1`）的概率，评估指标 ROC AUC。训练集 105,000 行、测试集 45,000 行，原始特征 56 列（13 数值 + 43 类别）。

**最终提交**：[submissions/20261016_FINAL_full_Bseeds.csv](submissions/20261016_FINAL_full_Bseeds.csv)，公开榜 **0.92301**（起点 0.91857）。全部实验数字见 [RESULTS.md](RESULTS.md)，提交记录见 [submissions/README.md](submissions/README.md)。

## 主线流程

|步骤|notebook|做什么|结果目录|运行环境|
|---|---|---|---|---|
|1|[01_hmda_data_preparation](01_hmda_data_preparation.ipynb)|审计原始 CSV，生成固定的 56 列特征、标签、ID、seed=42 五折和哈希清单|`artifacts/prepared/`|CPU|
|2|[02_hmda_models](02_hmda_models.ipynb)|基线：LR、XGBoost、CatBoost R1/R2 五折比较（CatBoost R2 = 0.91604）|`artifacts/models/`|CPU|
|3|[09_hmda_colab_gpu_catboost_tuning](09_hmda_colab_gpu_catboost_tuning.ipynb)|CatBoost 调参：内层早停 + Bernoulli 抽样，选出 C7 / C11|`artifacts/gpu_tuning/catboost_screen_v1/`|Colab T4|
|4|[10_hmda_colab_gpu_feature_ablation](10_hmda_colab_gpu_feature_ablation.ipynb)|第一轮特征消融：LTV 重算与差值等（+0.0030）|`artifacts/feature_ablation/feature_blocks_v1/`|Colab T4|
|5|[12_hmda_colab_gpu_feature_round2](12_hmda_colab_gpu_feature_round2.ipynb)|第二轮特征消融：LTV 申报精度、LTV 产品门槛等（+0.0013）|`artifacts/feature_ablation/feature_blocks_round2_v1/`|Colab T4|
|6|[13_hmda_colab_gpu_round2_final](13_hmda_colab_gpu_round2_final.ipynb)|最佳参数 × 最佳特征的五折模型 M0–M3 及三组融合（0.92051）|`artifacts/final_gpu/round2_final_v1/`|Colab T4|
|7|[15_hmda_colab_final_ensemble](15_hmda_colab_final_ensemble.ipynb)|M1/M2/M3 各加 2 个种子（0.92073）；检查 LightGBM 融合（无帮助）|同上|Colab T4|
|8|[17_hmda_colab_gpu_full_refit](17_hmda_colab_gpu_full_refit.ipynb)|学习曲线（多 25% 数据 +0.0012）+ M1/M2/M3 × 3 种子全部 105,000 行重训 → **最终提交**|`artifacts/final_gpu/full_refit_v1/`|Colab T4|

步骤 3–5 是选择参数和特征的依据；**复现最终提交只需要 01 → 13 → 15 → 17**（15 复用 13 的结果目录，17 读取 13/15 的五折轮数）。

最终模型：CatBoost（GPU）三组配置 × 3 个随机种子，共 9 个模型，每个都在全部训练数据上训练，测试概率先按组平均种子、再三组等权平均。

|组|参数|特征|
|---|---|---|
|M1|C11：depth 6、lr 0.03、L2=10、Bernoulli 0.8|G7：原始 56 列 + DTI 上下界、LTV 重算、收入比值、缺失组合、LTV 申报精度、LTV 产品门槛、反推房价、期限（72 列）|
|M2|C7：depth 8、lr 0.03、L2=10、Bernoulli 0.8|G7|
|M3|C11|原始 56 列 + DTI、LTV 重算、收入比值、缺失组合、LTV 精度、LTV 门槛（68 列）|

## 补充实验（结论都已写入 RESULTS.md，没有进入最终模型）

|notebook|内容|结论|
|---|---|---|
|[07_hmda_transformations_comparison](07_hmda_transformations_comparison.ipynb)|Ridge / RandomForest 上比较二阶、三阶、RBF、样条、分位数数值变换|线性模型最多 +0.003，仍远低于 CatBoost|
|[14_hmda_colab_gpu_feature_round3](14_hmda_colab_gpu_feature_round3.ipynb)|第三轮特征：产品标志组合、LTV 区间一致性、风险叠加、相对普查区收入|均未达标，特征已饱和|
|[16_hmda_colab_gpu_tabm_blend](16_hmda_colab_gpu_tabm_blend.ipynb)|神经网络 TabM-mini 与 CatBoost 融合（第 1 折）|单独弱 0.006，与 CatBoost 高度相关，融合无帮助|
|[18_hmda_colab_gpu_pseudo_label](18_hmda_colab_gpu_pseudo_label.ipynb)|测试集软伪标签，逐折诚实验证|+0.00008，无效|
|[19_hmda_colab_gpu_data_efficiency](19_hmda_colab_gpu_data_efficiency.ipynb)|CatBoost 训练方式：有序提升、Depthwise/Lossguide、Poisson、分箱、类别组合等|现有设置已是最好|

编号保留实验先后顺序。03、04、05、08、11 已被后续 notebook 取代并删除（其数字保留在 RESULTS.md 和提交记录中），06 从未存在。

## 运行方式

- **Colab notebook（09 及以后）**：整个项目文件夹放在 Google Drive 的 `Colab Notebooks/` 下，打开 notebook，运行时类型选 **T4 GPU**，「全部运行」。每个模型训练完立即写入 Drive，断线后重新「全部运行」会跳过已完成部分。依赖在 notebook 内自动安装（CatBoost 固定 1.2.10）。
- **01、02、07**：在项目目录启动 Jupyter（依赖见 `requirements.txt`），先运行 01。02 默认 `review` 只校验数据；训练需把顶部的 `NOTEBOOK_RUN_MODE` 改为 `full`（或设置环境变量 `HMDA_RUN_MODE=full`、`HMDA_MODELS=...`）。
- 所有 notebook 都会先核对 `artifacts/prepared/` 的哈希与固定五折，数据不一致时直接报错。

## 目录

```
train.csv / test.csv / sample_submission.csv   原始数据（只读）
01 … 19 *.ipynb                                 主线与补充实验（见上表）
artifacts/
  prepared/                                     01：固定特征、标签、五折、哈希
  models/                                       02：基线五折结果
  gpu_tuning/catboost_screen_v1/                09：调参
  gpu_tuning/data_efficiency_screen_v1/         19：训练方式筛选
  feature_ablation/feature_blocks_v1/           10：第一轮特征
  feature_ablation/feature_blocks_round2_v1/    12：第二轮特征
  feature_ablation/feature_blocks_round3_v1/    14：第三轮特征
  final_gpu/round2_final_v1/                    13、15：五折模型、OOF、测试预测
  final_gpu/full_refit_v1/                      17：学习曲线、全量模型、最终提交
  final_gpu/pseudo_label_v1/                    18：伪标签验证
  nn_models/tabm_mini_v1/                       16：TabM-mini
  transformation_models/                        07：数值变换比较
submissions/                                    最终提交文件与完整提交记录
tools/                                          生成 09–19 notebook 的脚本（见 tools/README.md）
RESULTS.md                                      全部实验结果
```
