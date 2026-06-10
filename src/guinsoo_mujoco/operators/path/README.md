# 路径算子

## 弧长参数化跟踪

沿密化路径以速度 \(v\) 推进弧长 \(s\)，参考关节

\[
q_{ref}(s) = \text{lerp}(q_i, q_{i+1})
\]

PD 控制：

\[
u = K_p (q_{ref} - q) - K_d \dot{q}
\]

## 路径密化

对 RRT 折线路径按最大关节步长插值，每中间点经 `CollisionModel` 检验。

## API

- `JointPathTracker`
- `densify_path(runtime, path, collision_model, max_joint_step=...)`
- `snap_path_start(path, q_start)`
