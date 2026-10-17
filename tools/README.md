# tools

- `build_notebooks.py`：生成项目的 5 本 notebook。
- `cells/*.py`：notebook 中的训练代码，每个文件是一个代码单元格。它们抽取自在 Colab 上跑过的实验 01、09–19。

用法：

```bash
python tools/build_notebooks.py <输出目录>     # 例如 python tools/build_notebooks.py /tmp/nb
```

`cells/` 的文件名前缀：

|前缀|来源|
|---|---|
|`data_contract`、`helpers`、`dti_bounds`、`features_round1`–`3`|多个 notebook 共用：核对固定数据接口、通用函数、DTI 上下界、三轮特征|
|`prep_`|实验 01（01 的数据审计与固定数据接口）|
|`ablation_`|实验 10、12、14（02 的三轮特征消融）|
|`tuning_`、`screen_`|实验 09、19（03 的调参与训练方式筛选）|
|`cv_`、`refit_`|实验 13、15、17（04 的五折模型、学习曲线与全量重训）|
|`lgbm_`、`tabm_`、`pseudo_`|实验 15、16、18（05 的 LightGBM、TabM-mini、软伪标签）|

notebook 中每个实验的结果目录改成了 `artifacts/<notebook>/<子目录>`（对照见项目 README）；原始 notebook 和生成它们的脚本在 Git 历史的第一个提交中。

项目根目录里的 notebook 在 Colab 上运行后会带有输出。直接输出到项目根目录会用无输出的版本覆盖它们，所以修改时先输出到其他目录，确认后再复制。
