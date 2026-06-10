# Guinsoo MuJoCo Platform

Guinsoo MuJoCo Platform 是一款面向 Mac 的 MuJoCo 运动控制算法验证平台。第一版目标是提供一个原生 Qt 桌面工具：选择机器人示例，运行 MuJoCo 仿真，在内嵌 OpenGL 视图中观察运动，查看控制曲线，采集 episode 数据，并在本地 notebook 中分析数据。

## 第一版能力

- 原生桌面界面：基于 PySide6 的 `Guinsoo Sim Studio`。
- 仿真核心：`MuJoCoRuntime` 负责模型加载、仿真步进、状态读取和控制写入。
- 机器人注册表：内置 UR5e、OpenArm、XLeRobot 风格机器人三个适配入口。
- 算法示例：第一版以经典关节位置控制和 IK/到点类 demo 为主。
- 数据采集：`RunRecorder` 输出 HDF5 数据和 JSON 元数据。
- 数据回放：`RunBrowser` 可以读取 episode，用于桌面回放和 notebook 分析。
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

## 代码结构

- `src/guinsoo_mujoco/robots.py`：机器人注册表和适配器定义。
- `src/guinsoo_mujoco/runtime.py`：MuJoCo 运行时封装。
- `src/guinsoo_mujoco/controllers.py`：控制器接口和经典控制示例。
- `src/guinsoo_mujoco/data.py`：HDF5 记录和 episode 回放。
- `src/guinsoo_mujoco/app/`：PySide6 桌面界面。
- `assets/robots/`：机器人资产 manifest。
- `docs/`：中文设计、数据格式和验收说明。
- `notebooks/`：本地数据分析入口。

## 第一版边界

第一版聚焦桌面工具和端到端样板，不把以下内容作为阻塞项：实时外部流、ROS2/MCAP 主链路、`.app` 打包、完整 XLeRobot 高保真模型、RL 或模仿学习训练。
