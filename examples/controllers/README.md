# 控制器示例说明

第一版内置控制器以经典控制为主，重点是给算法复现和平台验证提供稳定基线。

## 已实现示例

- `JointPositionController`：关节位置 PD 控制器。

## 控制器合约

控制器需要实现两个方法：

- `reset(runtime, config)`：重置内部状态，可读取配置。
- `step(runtime, t, dt)`：读取运行时状态，写入控制量，并返回一份可记录的诊断数据。

后续 IK、轨迹跟踪和抓取到点示例应沿用同一合约。

独立 Demo 包（含场景、规划与控制）见 `src/guinsoo_mujoco/demos/`。例如 UR5e 末端避障到点 demo 在 `demos/ur5e/ee_pose_avoid/`，通过 `DemoRegistry` 注册，不依赖 Qt 界面。
