# 控制器示例说明

## 控制器合约

`guinsoo_mujoco.controllers` 定义 `Controller` / `RuntimeLike` 协议：

- `reset(runtime, config)`：重置内部状态
- `step(runtime, t, dt)`：写入控制量并返回诊断数据

## 核心算子

具体控制与规划实现位于 `src/guinsoo_mujoco/operators/`：

- `operators.control`：`JointPositionController`、`ReachMotionController`
- `operators.ik`、`operators.rrt`、`operators.path`、`operators.collision`

## Demo 包

每个 Demo 在 `src/guinsoo_mujoco/demos/{robot}/{demo_id}/` 独立目录，包含 `controller.py`、`config.py`、`DESIGN.md`，通过 `DemoRegistry` 注册。

示例：UR5e 末端避障到点 → `demos/ur5e/ee_pose_avoid/`。
