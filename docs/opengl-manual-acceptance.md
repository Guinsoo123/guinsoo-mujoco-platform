# macOS OpenGL 手动验收

PySide6 的 OpenGL 视图需要在真实 macOS 图形会话中验收，不能只依赖无头自动测试。

## 验收步骤

1. 激活环境：`conda activate robot_dev`。
2. 启动界面：`python -m guinsoo_mujoco.app.main`。
3. 确认主窗口标题为 `Guinsoo Sim Studio`。
4. 左侧可以看到 UR5e、OpenArm、XLeRobot 三个机器人入口。
5. 中间 OpenGL 视图不为空，切换机器人时能更新状态文本。
6. 点击“运行”后底部曲线持续刷新。
7. 点击“暂停”和“重置”后界面状态正确变化。

## 后续高保真验收

接入真实 MuJoCo 模型后，需要补充：

- UR5e 模型加载无 XML 或 mesh 错误。
- 相机视角、关节运动和控制曲线同步。
- 记录 episode 后可回放并被 notebook 读取。
