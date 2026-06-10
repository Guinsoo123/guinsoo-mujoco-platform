# 数据格式说明

平台使用 HDF5 作为 episode 主数据格式，并为每个数据包生成同名 JSON 元数据，以及可直接导入 PlotJuggler 的 CSV 时间序列。

## 默认保存目录

```
~/.guinsoo_mujoco/runs/
```

每次停止记录会生成三个文件：

- `YYYYMMDD-HHMMSS-<robot>-<demo>.h5`
- `YYYYMMDD-HHMMSS-<robot>-<demo>.json`
- `YYYYMMDD-HHMMSS-<robot>-<demo>_plotjuggler.csv`

## HDF5 数据集

| 数据集 | 含义 |
|--------|------|
| `time` | 仿真时间戳 |
| `qpos` | 关节位置反馈 |
| `qvel` | 关节速度反馈 |
| `target` | 控制器目标位置（指令值） |
| `ctrl` | 写入 MuJoCo 的控制量 |
| `actuator_force` | 执行器输出力/力矩 |
| `qfrc_actuator` | 关节空间执行器力矩 |
| `sensors/control_command` | 控制器输出的控制命令 |
| `sensors/<name>` | 其它算法诊断量 |

## JSON 元数据

除机器人、控制器、MuJoCo 版本、资产来源、应用版本、运行配置外，还包含：

- `demo`
- `joint_names`
- `actuator_names`
- `created_at`
- `hdf5_file`
- `plotjuggler_csv`

## PlotJuggler CSV 列命名

CSV 使用扁平列名，便于 PlotJuggler 按层级浏览，例如：

- `timestamp`
- `joint/shoulder_pan/qpos`
- `joint/shoulder_pan/target`
- `joint/shoulder_pan/ctrl`
- `joint/shoulder_pan/actuator_force`
- `joint/shoulder_pan/qfrc_actuator`
- `joint/shoulder_pan/tracking_error`

## 读取示例

```python
from guinsoo_mujoco.data import RunBrowser

episode = RunBrowser.open(("~/.guinsoo_mujoco/runs/example.h5", "~/.guinsoo_mujoco/runs/example.json"))
print(episode.metadata)
print(episode.qpos.shape)
print(episode.actuator_force.shape)
```

## 重新导出 PlotJuggler CSV

```bash
python -m guinsoo_mujoco.cli export-plotjuggler ~/.guinsoo_mujoco/runs/<episode>.h5
```
