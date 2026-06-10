# 碰撞检测算子

对关节构型 \(q\) 做可行性检验，供 IK、RRT、路径密化复用。

## 距离判定

对机械臂 geom 与障碍 geom 调用 `mj_geomDistance`，若

\[
d(g_i, g_j) < m
\]

则判为碰撞，其中 \(m\) 为 `CollisionModel.margin`。

## 接触补充

同时检查 `data.contact[].dist < margin` 的 robot–obstacle 接触对。

## API

- `CollisionModel`：由 demo `config` 注入 robot/ignore body 名与障碍 geom 名
- `is_configuration_colliding(runtime, qpos, model)`
- `is_edge_colliding(runtime, q_from, q_to, model, samples=...)`
