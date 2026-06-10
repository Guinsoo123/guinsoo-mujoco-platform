# Guinsoo MuJoCo Platform v1 中文设计方案

## 概要

本平台面向 Mac 本地运动控制算法验证，技术栈为 Conda、Python、PySide6 和 MuJoCo。第一版优先交付原生 Qt 桌面可视化工具，让用户可以选择机器人示例、运行仿真、查看内嵌 OpenGL 视图和实时曲线、采集 episode 数据，并通过本地 notebook 分析。

第一版机器人范围采用“一稳两预览”：

- 稳定样板：UR5e，来源优先使用 `google-deepmind/mujoco_menagerie`。
- 预览样板：OpenArm，参考 `enactic/openarm` 和 `enactic/dora-openarm-mujoco`。
- 实验性预览：XLeRobot 风格轮式双臂机器人，接入前需要完成许可证和模型质量检查。

## 核心架构

平台采用 PySide6 单进程桌面架构。主界面是 Sim Studio，左侧选择机器人和 demo，中间是 MuJoCo OpenGL 视图，右侧是运行控制和参数，底部显示实时曲线。

核心模块分层如下：

- `RobotRegistry` 管理机器人适配器、支持等级、manifest 路径和 demo 列表。
- `MuJoCoRuntime` 负责加载 XML 模型、推进仿真、读取状态和写入控制量。
- `Controller` 定义控制器 step 合约，第一版内置经典关节位置控制。
- `RunRecorder` 写入 HDF5 数据和 JSON 元数据。
- `RunBrowser` 读取 episode，服务回放、曲线查看和 notebook 分析。
- `AssetDownloader` 根据 manifest 把第三方模型下载到本地缓存。

## 数据与资产

数据主格式为 HDF5，配套 JSON 元数据记录机器人、控制器、MuJoCo 版本、资产来源、应用版本和运行配置。HDF5 内包含 `time`、`qpos`、`qvel`、`ctrl` 和 `sensors/*`。

机器人模型资产不直接提交到仓库。仓库只保存 manifest、来源链接、许可证信息、缓存目录和入口文件。UR5e 可自动下载；许可证未知的实验性来源必须人工复核后再启用下载。

## 验收标准

- Apple Silicon Mac 上可以通过 `environment.yml` 创建或更新 `robot_dev` 环境。
- PySide6 App 可以启动主界面。
- UR5e 适配器作为稳定样板注册，支持经典控制和 IK demo 入口。
- OpenArm 与 XLeRobot 作为预览适配器注册，并在界面中明确显示支持等级。
- HDF5 数据包可由 `RunBrowser` 和 notebook 读取。
- 自动测试覆盖 registry、controller step 合约、recorder schema、asset manifest 解析和 App 模型。
- macOS OpenGL 内嵌渲染通过手动验收。

## 默认取舍

第一版选择原生 Qt App，不做本地 Web 控制台。第一版采用 PySide6、单进程、内嵌 OpenGL。算法示例以经典控制和 IK 为主，暂不做 RL 或模仿学习训练。数据主格式为 HDF5 加 JSON 元数据，暂不做 ROS2/MCAP 主链路。
