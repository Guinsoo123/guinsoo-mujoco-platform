# Notebook 分析入口

本目录保存面向 Mac 本地分析的 notebook 示例。第一版目标是让用户可以读取 HDF5 episode、查看状态曲线、对比控制输入，并把结果用于算法复现报告。

推荐在 `robot_dev` 环境中启动：

```bash
conda activate robot_dev
jupyter lab notebooks/
```

如果环境里没有 Jupyter：

```bash
conda install -n robot_dev -c conda-forge jupyterlab matplotlib pandas
```
