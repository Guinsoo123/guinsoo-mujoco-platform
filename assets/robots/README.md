# 机器人资产说明

本目录只保存机器人资产 manifest，不直接提交大型第三方 mesh、纹理或完整上游模型目录。

## 支持等级

- `stable`：第一版稳定样板，要求能完整进入加载、运行、记录和回放闭环。
- `preview`：预览适配器，目标是验证平台接口和模型扩展方式。
- `experimental`：实验性预览，来源、许可证或模型质量仍需人工复核。

## 现有 manifest

- `ur5e.json`：Universal Robots UR5e，来源为 DeepMind MuJoCo Menagerie。
- `openarm.json`：OpenArm 预览入口，参考 enactic 开源项目。
- `xlerobot.json`：XLeRobot 风格轮式双臂机器人实验入口，当前禁止自动下载。

## 下载方式

```bash
conda run -n robot_dev python -m guinsoo_mujoco.cli fetch-assets ur5e
```

默认缓存目录为 `~/.guinsoo_mujoco/assets`。
