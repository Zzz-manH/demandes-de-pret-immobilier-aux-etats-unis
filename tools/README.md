# tools

- `build_notebooks.py`：生成 Colab notebook 09–19 的脚本（notebook 中的代码都由它产生）。脚本里也保留了已删除的 08、11 的生成代码，便于追溯。
- `shared/data_contract_cell.py`、`shared/dti_bounds_cell.py`：多本 notebook 共用的两个单元格（数据契约核对、DTI 上下界），原来位于已删除的 04。

用法：

```bash
python tools/build_notebooks.py <输出目录> [编号 ...]    # 例如 python tools/build_notebooks.py /tmp/nb 17
```

项目根目录里的 notebook 是在 Colab 上**跑过、带输出**的版本。直接输出到项目根目录会用无输出的版本覆盖它们，所以修改时先输出到其他目录，确认后再复制。2026-10-16 核对过：本脚本生成的 09、10、12–19 与项目中这些 notebook 的每个单元格源码完全一致。
