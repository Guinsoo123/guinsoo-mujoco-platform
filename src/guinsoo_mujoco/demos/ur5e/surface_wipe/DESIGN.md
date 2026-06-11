# UR5e 曲面擦拭 Demo（法向导纳 + 切向轨迹）

## 目标

在正弦波浪面上沿 **+X** 方向擦拭，同时通过 **法向二阶导纳** 维持约恒定的接触力。

## 控制结构

```text
切向：s += v_t * dt，参考点 p_ref(s) 来自 SineSheetSurface
法向：F_n → 导纳(M,B,K) → d_n → p_d = p_ref + (standoff + d_n) * n
内环：IK(attachment_site) → actuator_joint_target → ctrl
```

法向动力学：

\[
M \ddot{d}_n + B \dot{d}_n + K d_n = F_n - F_{des}
\]

## 场景

- 波浪面：`hfield` 近似 \(z = z_0 + A\sin(2\pi x/\lambda)\)
- 力传感器：`tool_force` / `tool_torque` @ `attachment_site`
- 路径：\(s \in [0, 0.25]\) m，\(x\) 从 0.30 到 0.55

## FSM

| 阶段 | 行为 |
|------|------|
| APPROACH | IK 到路径起点上方 |
| DESCEND | 沿 -n 下降直至接触力阈值 |
| FOLLOW | 切向推进 + 法向导纳 |
| RETRACT | 沿 +n 抬离 |
| DONE | 保持 |

## 默认参数（config.py）

| 参数 | 值 |
|------|-----|
| F_des | 15 N |
| M, B, K | 1, 80, 0 |
| v_t | 0.05 m/s |
| WIPE_LENGTH | 0.25 m |
| amplitude / wavelength | 0.015 m / 0.12 m |

## Sim Studio 验收

1. 选择 **UR5e → 曲面擦拭 (导纳)**
2. 运行并开始记录
3. 观察末端沿波浪面 +X 移动并贴附
4. episode 中检查 `sensor/path_s` 增至 0.25，`sensor/force_normal` 在 F_des 附近

## 实现说明

- 力传感器读数为 **site 坐标系**，控制器变换到世界系后取 **压向曲面为正** 的法向力
- 工作台 `table` 已移除；波浪面由 **`wave_stand` 窄支架**（纯视觉）支撑，不占用机械臂工作空间
- `<contact><exclude>` 禁止肩/前臂等与波浪面碰撞，仅腕部 `eef_collision` 可接触
- HOME 构型由 IK 预解至路径起点上方，保证 APPROACH 一步到位
- `attachment_site` 的 **+Z 轴** 与曲面法向 `n` 对齐（Menagerie 工具坐标约定）
- FOLLOW 使用 **6DoF IK**（`position_only=False`），姿态误差通常 < 1°
- 法向导纳偏移 + 切向 `path_s` 在 IK 成功时匀速推进
- episode 标量传感器（`path_s`、`force_normal` 等）导出为 `sensor/<name>` 单列

## 调参建议

- 力偏低：略增 `F_des` 或减小 `CONTACT_STANDOFF`
- 力控振荡：增大 `damping`、降低 `force_lpf_alpha`、检查 wave `solref`
- 贴附不足：降低 `TANGENTIAL_SPEED`，检查 `ee_pos_error` 是否 < 1 cm
