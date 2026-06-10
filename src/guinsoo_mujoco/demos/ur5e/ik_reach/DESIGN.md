# UR5e 关节正弦运动 Demo

## 场景与目标

在 home 构型附近，对**单个关节**施加正弦轨迹扰动，用于观察关节伺服响应。

> **注意**：本 Demo 名为 `ik_reach` 但**不使用** `operators.ik` 笛卡尔 IK；仅做关节空间正弦运动。

## 依赖算子

- `operators.control.ReachMotionController`

## Demo 流程

每步更新目标关节：

\[
q_{target,i} = q_{home,i} + A \sin(2\pi f t), \quad i = \text{joint\_index}
\]

再经 PD 写入 `ctrl`。

## 数学模型

\[
u = K_p (q_{target} - q) - K_d \dot{q}
\]

参数见 [`config.py`](config.py)。

## 文件映射

| 文件 | 职责 |
|------|------|
| `config.py` | 振幅、频率、关节索引 |
| `controller.py` | 实例化 `ReachMotionController` |

## 运行与验证

Sim Studio：UR5e → **关节正弦运动** → 运行。
