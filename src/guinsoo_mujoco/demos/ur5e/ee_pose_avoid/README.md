# UR5e 末端避障到点 Demo

## 场景说明

本 Demo 在 Menagerie UR5e 机械臂工作空间中布置：

- **工作台** `table`：静态盒体障碍
- **三个障碍柱** `obstacle_a/b/c`：阻挡直线路径，迫使规划绕障
- **四个目标位姿标记**（mocap 坐标轴）：仅用于可视化，不参与碰撞

```
        [over_obstacle]  (0.55, 0, 0.55)
              |
    [home] ---+--- [obstacles]
              |
         [approach]        [place]
```

| 路点 | 名称 | 位置 (x,y,z) | 说明 |
|------|------|--------------|------|
| 1 | home | 0.35, 0, 0.45 | 工作台上方安全位 |
| 2 | approach | 0.45, -0.25, 0.35 | 障碍前方接近位 |
| 3 | over_obstacle | 0.55, 0, 0.55 | 抬高绕过障碍 |
| 4 | place | 0.40, 0.25, 0.30 | 最终放置位 |

路点定义见 [`config.py`](config.py) 中的 `WAYPOINTS`。

## 整体流水线

```
目标位姿 → 数值 IK 求 q_goal → RRT-Connect 规划无碰撞关节路径 → 关节 PD 跟踪
```

每个路点循环执行 **Plan → Track → Hold** 状态机：

1. **Plan**：对当前路点做 IK，再用 RRT-Connect 从 `q_start` 规划到 `q_goal`
2. **Track**：沿路径做时间参数化插值，PD 控制各关节跟踪
3. **Hold**：到达后保持约 1 秒，切换下一路点

## 数值 IK（阻尼最小二乘）

末端使用 Menagerie 的 `attachment_site`。

位姿误差：

- 位置：\( e_p = p_{target} - p_{current} \)
- 姿态：\( e_R = \text{axis-angle}(R_{target} R_{current}^T) \)

迭代更新：

\[
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} e, \quad q \leftarrow \text{clip}(q + \dot{q})
\]

实现见 [`ik.py`](ik.py)。

## RRT-Connect

在 6 维关节空间采样，以 MuJoCo 碰撞检测作为可行性检验。

伪代码：

```
tree_a ← {q_start}, tree_b ← {q_goal}
repeat max_iterations:
    q_rand ← sample with goal_bias toward opposite tree root
    q_new ← steer(nearest(tree_a, q_rand))
    if collision_free(q_new):
        add q_new to tree_a
        try connect to nearest node in tree_b
        if connected: return path
    swap tree_a, tree_b
```

关键参数（[`config.py`](config.py)）：

- `RRT_STEP_SIZE = 0.15` rad
- `RRT_GOAL_BIAS = 0.25`
- `RRT_MAX_ITERATIONS = 3000`

实现见 [`rrt.py`](rrt.py)。

## 碰撞检测

对候选关节构型 `q`：

1. 写入 `data.qpos`，调用 `mj_forward`
2. 检查 `data.contact` 中是否出现障碍物 geom
3. 对障碍物与机械臂 geom 调用 `mj_geomDistance` 作为补充

实现见 [`collision.py`](collision.py)。

### Python → C++ MuJoCo API 对照

| Python | C++ |
|--------|-----|
| `mujoco.mj_forward(m, d)` | `mj_forward(m, d)` |
| `mujoco.mj_jacSite(m, d, jacp, jacr, site_id)` | `mj_jacSite(m, d, jacp, jacr, site_id)` |
| `mujoco.mj_geomDistance(...)` | `mj_geomDistance(...)` |
| `data.ncon`, `data.contact[i].dist` | `d->ncon`, `d->contact[i].dist` |
| `data.site_xpos`, `data.site_xmat` | `d->site_xpos`, `d->site_xmat` |

## 路径跟踪

将 RRT 路径节点序列按弧长参数化，以速度 `PATH_SPEED` 推进，输出关节目标：

\[
u = k_p (q_{ref} - q) - k_d \dot{q}
\]

与平台内置 `JointPositionController` 使用相同 PD 结构，便于替换和对比。

实现见 [`path_tracker.py`](path_tracker.py) 与 [`controller.py`](controller.py)。

## 文件清单（移植时复制）

| 文件 | 职责 |
|------|------|
| `config.py` | 路点、障碍名、增益与规划参数 |
| `ik.py` | 数值 IK |
| `collision.py` | 构型碰撞检测 |
| `rrt.py` | RRT-Connect 规划器 |
| `path_tracker.py` | 关节路径跟踪 |
| `controller.py` | Plan/Track/Hold 状态机 |
| `scene.xml` | 障碍物与目标标记场景 |

**不依赖 PySide6**。对外只需：

- 输入：`MuJoCoRuntime`（或等价的 `mjModel`/`mjData`）
- 输出：每步 `control` 向量写入 `data.ctrl`

## C++ 移植步骤建议

1. 复制算法文件逻辑，将 NumPy 矩阵运算改为 Eigen
2. 用 `mj_name2id` 解析 `attachment_site` 与障碍物 geom
3. 在独立线程或控制周期开始时运行 RRT（规划耗时与实时步进分离）
4. 跟踪阶段复用现有关节伺服接口
5. 场景 MJCF 中保留相同路点坐标，或改为从配置文件加载

## 运行方式

1. 下载 UR5e 资产：`python -m guinsoo_mujoco.cli fetch-assets ur5e`
2. 启动 Sim Studio，左侧选择 **UR5e** → **末端避障到点 (RRT)**
3. 点击 **运行** 观察依次到点与绕障路径
