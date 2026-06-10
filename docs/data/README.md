# 数据格式说明

平台第一版使用 HDF5 作为 episode 主数据格式，并为每个数据包生成同名 JSON 元数据。

## HDF5 数据集

- `time`：每帧仿真时间。
- `qpos`：MuJoCo 位置状态。
- `qvel`：MuJoCo 速度状态。
- `ctrl`：控制输入。
- `sensors/<name>`：可选传感器或算法诊断量，例如末端误差。

## JSON 元数据

元数据包含：

- `robot_id`
- `controller`
- `mujoco_version`
- `asset_source`
- `app_version`
- `config`
- `created_at`
- `hdf5_file`

## 读取示例

```python
from guinsoo_mujoco.data import RunBrowser

episode = RunBrowser.open(("data/runs/example.h5", "data/runs/example.json"))
print(episode.metadata)
print(episode.qpos.shape)
```
