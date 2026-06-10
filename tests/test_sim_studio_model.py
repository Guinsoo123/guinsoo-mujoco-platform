from guinsoo_mujoco.app.model import SimStudioModel


def test_sim_studio_model_lists_robot_cards_for_main_window():
    model = SimStudioModel()

    cards = model.robot_cards()

    assert cards[0]["robot_id"] == "ur5e"
    assert cards[0]["support_label"] == "稳定样板"
    assert cards[1]["support_label"] == "预览适配"
    assert cards[2]["support_label"] == "实验性预览"


def test_sim_studio_model_selects_robot_and_demo():
    model = SimStudioModel()

    selection = model.select("ur5e", "ik_reach")

    assert selection.robot.robot_id == "ur5e"
    assert selection.demo == "ik_reach"


def test_sim_studio_model_lists_demo_cards_for_ur5e():
    model = SimStudioModel()

    cards = model.demo_cards("ur5e")
    demo_ids = [card["demo_id"] for card in cards]

    assert "ee_pose_avoid" in demo_ids
    assert any(card["display_name"] == "末端避障到点 (RRT)" for card in cards)
