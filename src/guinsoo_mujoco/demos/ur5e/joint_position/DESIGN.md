# UR5e 关节位置保持 Demo

## 场景与目标

使用 Menagerie 默认 UR5e 场景，机械臂保持固定关节 home 构型。

## 依赖算子

- `operators.control.JointPositionController`

## Demo 流程

每仿真步读取 \((q, \dot{q})\)，计算 PD 控制量并写入 `data.ctrl`。

## 数学模型

\[
u = K_p (q_{target} - q) - K_d \dot{q}
\]

参数见 [`config.py`](config.py)：`KP=20`，`KD=2`。

## 文件映射

| 文件 | 职责 |
|------|------|
| `config.py` | 增益与 home 构型 |
| `controller.py` | 实例化 `JointPositionController` |

## 运行与验证

Sim Studio：UR5e → **关节位置保持** → 运行。
