from guinsoo_mujoco.plotjuggler_analysis import (
    build_motion_control_layout,
    find_plotjuggler_executable,
)


def test_build_motion_control_layout_uses_six_joint_panels_per_tab():
    joints = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    layout = build_motion_control_layout(joints)

    assert layout.count("<Tab ") == 5
    assert layout.count("<DockArea ") == 30
    assert 'DockArea name="shoulder_pan"' in layout
    assert 'DockArea name="wrist_3"' in layout
    assert layout.count('orientation="-"') == 5
    assert layout.count('orientation="|"') == 10
    assert "joint/shoulder_pan/qpos" in layout
    assert "joint/shoulder_pan/target" in layout
    assert "joint/elbow/qvel" in layout
    assert "joint/wrist_2/qfrc_actuator" in layout


def test_build_motion_control_layout_keeps_one_signal_per_velocity_panel():
    layout = build_motion_control_layout(["shoulder_pan", "elbow"])

    shoulder_block = layout.split('DockArea name="shoulder_pan"')[1].split("</DockArea>")[0]
    assert shoulder_block.count("<curve ") == 2
    assert "joint/shoulder_pan/qpos" in shoulder_block
    assert "joint/shoulder_pan/target" in shoulder_block

    velocity_tab = layout.split('tab_name="4_关节速度"')[1].split("</Tab>")[0]
    assert velocity_tab.count('DockArea name="shoulder_pan"') == 1
    assert velocity_tab.count("joint/shoulder_pan/qvel") == 1
    assert velocity_tab.count("joint/elbow/qvel") == 1


def test_find_plotjuggler_executable_or_skip():
    try:
        path = find_plotjuggler_executable()
    except FileNotFoundError:
        return
    assert path.name == "PlotJuggler"
