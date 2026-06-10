# XLeRobot 预览运动 Demo

## 场景与目标

以较低 PD 增益保持关节 home，便于在 Sim Studio 中预览 XLeRobot 模型。

## 依赖算子

- `operators.control.JointPositionController`（`KP=12`, `KD=1.5`）

## 数学模型

\[
u = K_p (q_{target} - q) - K_d \dot{q}
\]

## 文件映射

| 文件 | 职责 |
|------|------|
| `config.py` | 预览用增益 |
| `controller.py` | 控制器工厂 |

## 运行与验证

Sim Studio：XLeRobot → **预览运动** → 运行。
