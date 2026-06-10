# Guinsoo MuJoCo Platform

Guinsoo MuJoCo Platform 是一款面向 Mac 的 MuJoCo 运动控制算法验证平台。第一版目标是提供一个原生 Qt 桌面工具：选择机器人示例，运行 MuJoCo 仿真，在内嵌 OpenGL 视图中观察运动，查看控制曲线，采集 episode 数据，并在本地 notebook 中分析数据。

## 第一版能力

- 原生桌面界面：基于 PySide6 的 `Guinsoo Sim Studio`。
- 仿真核心：`MuJoCoRuntime` 负责模型加载、仿真步进、状态读取和控制写入。
- 机器人注册表：内置 UR5e、OpenArm、XLeRobot 风格机器人三个适配入口。
- 算法示例：第一版以经典关节位置控制和 IK/到点类 demo 为主。
- 数据采集：`RunRecorder` 输出 HDF5、JSON 元数据和 PlotJuggler CSV。
- 数据监测：记录关节位置/速度反馈、目标指令、控制量、执行器力矩与跟踪误差。
- 数据回放：`RunBrowser` 可以读取 episode，用于桌面回放、PlotJuggler 和 notebook 分析。
- 资产管理：仓库保存 manifest 和适配代码，第三方模型资产下载到本地缓存，不直接提交大型 mesh。

## 环境准备

推荐使用 Apple Silicon Mac 和 conda-forge。

```bash
conda env update -n robot_dev -f environment.yml
conda activate robot_dev
```

如果还没有 `robot_dev` 环境：

```bash
conda env create -f environment.yml
conda activate robot_dev
```

## 运行测试

```bash
conda run -n robot_dev python -m pytest tests
```

## 启动桌面工具

开发模式下可以直接运行：

```bash
conda run -n robot_dev python -m guinsoo_mujoco.app.main
```

安装为可编辑包后也可以运行：

```bash
conda run -n robot_dev guinsoo-sim-studio
```

## 下载机器人资产

UR5e 模型来源优先使用 DeepMind MuJoCo Menagerie。下载到默认本地缓存：

```bash
conda run -n robot_dev python -m guinsoo_mujoco.cli fetch-assets ur5e
```

XLeRobot 当前标记为实验性预览，manifest 中许可证为 `UNKNOWN`，下载器会拒绝自动拉取，直到完成来源和许可证检查。

## 记录与分析仿真数据

### 1. 在 App 中记录 episode

1. 启动 `Guinsoo Sim Studio` 并加载机器人场景。
2. 点击 **运行** 开始仿真。
3. 点击 **开始记录**，仿真运行期间会持续采样。
4. 点击 **停止并保存**，数据包写入默认目录。
5. 切换到右侧 **数据包** 标签页，可查看、删除录制结果，或一键启动 PlotJuggler 分析。

```bash
~/.guinsoo_mujoco/runs/
```

每个数据包包含：

| 文件 | 内容 |
|------|------|
| `*.h5` | 主时序数据（位置、速度、指令、力矩等） |
| `*.json` | 实验元数据（机器人、demo、关节名、配置） |
| `*_plotjuggler.csv` | 可直接导入 PlotJuggler 的扁平 CSV |

### 2. 用 PlotJuggler 一键分析（推荐）

1. 安装 [PlotJuggler](https://github.com/facontidavide/PlotJuggler) 到 `/Applications/PlotJuggler.app`。
2. 在 App 右侧 **数据包** 页选择 episode，点击 **PlotJuggler 分析**。
3. App 会自动加载 CSV，并套用标准运控分析布局。每个页面使用 **2×3 网格** 展示 6 个关节窗口，共 5 组图表：
   - 关节位置跟踪（每窗 `qpos` vs `target`）
   - 跟踪误差（每窗 `tracking_error`）
   - 控制力矩（每窗 `ctrl` vs `actuator_force`）
   - 关节速度（每窗 `qvel`）
   - 关节力矩反馈（每窗 `qfrc_actuator`）

也可手动打开 PlotJuggler，选择 **Data Load** → **CSV**，载入 `*_plotjuggler.csv`，将 `timestamp` 设为横轴。

若需要重新导出 CSV：

```bash
conda run -n robot_dev python -m guinsoo_mujoco.cli export-plotjuggler \
  ~/.guinsoo_mujoco/runs/<episode>.h5
```

### 3. 用 Python / Notebook 分析

```bash
conda activate robot_dev
jupyter lab notebooks/episode_analysis.ipynb
```

或在代码中读取：

```python
from guinsoo_mujoco.data import RunBrowser

episode = RunBrowser.open((
    "~/.guinsoo_mujoco/runs/<episode>.h5",
    "~/.guinsoo_mujoco/runs/<episode>.json",
))
print(episode.metadata["actuator_names"])
print(episode.target - episode.qpos)  # 跟踪误差
```

更完整的数据字段说明见 [docs/data/README.md](docs/data/README.md)。

## 代码结构

- `src/guinsoo_mujoco/robots.py`：机器人注册表和适配器定义。
- `src/guinsoo_mujoco/runtime.py`：MuJoCo 运行时封装。
- `src/guinsoo_mujoco/controllers.py`：控制器接口和经典控制示例。
- `src/guinsoo_mujoco/data.py`：HDF5 记录、PlotJuggler 导出和 episode 回放。
- `src/guinsoo_mujoco/recording.py`：App 侧记录会话与采样辅助。
- `src/guinsoo_mujoco/run_catalog.py`：episode 列表、删除与元数据浏览。
- `src/guinsoo_mujoco/plotjuggler_analysis.py`：PlotJuggler 启动与标准运控分析布局生成。
- `src/guinsoo_mujoco/app/runs_panel.py`：数据包管理界面。
- `src/guinsoo_mujoco/app/`：PySide6 桌面界面。
- `assets/robots/`：机器人资产 manifest。
- `docs/`：中文设计、数据格式和验收说明。
- `notebooks/`：本地数据分析入口。

## 第一版边界

第一版聚焦桌面工具和端到端样板，不把以下内容作为阻塞项：实时外部流、ROS2/MCAP 主链路、`.app` 打包、完整 XLeRobot 高保真模型、RL 或模仿学习训练。
