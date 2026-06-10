# 数值 IK 算子（阻尼最小二乘）

## 误差定义

- 位置：\( e_p = p_{target} - p_{current} \)
- 姿态：\( e_R = \text{axis-angle}(R_{target} R_{current}^T) \)

## 迭代更新

\[
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} e, \quad q \leftarrow \text{clip}(q + \dot{q})
\]

## API

- `IkOptions`：site 名、容差、阻尼、`CollisionModel`（可选）
- `solve_ik(runtime, target_pos, target_rot, q_init, options)`
- `solve_ik_multi_seed(runtime, target_pos, target_rot, seeds, options)`
