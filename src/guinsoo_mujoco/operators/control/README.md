# 关节 PD 控制算子

\[
u = K_p (q_{target} - q) - K_d \dot{q}
\]

## API

- `JointPositionController`：固定关节目标
- `ReachMotionController`：单关节正弦扰动（非笛卡尔 IK）
- `default_joint_target(dof)`：UR5e 默认 home 构型
