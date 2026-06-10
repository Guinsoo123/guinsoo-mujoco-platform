from guinsoo_mujoco.asset_downloader import parse_github_tree_url


def test_parse_github_tree_url_extracts_owner_repo_branch_and_path():
    parsed = parse_github_tree_url(
        "https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e"
    )

    assert parsed.owner == "google-deepmind"
    assert parsed.repo == "mujoco_menagerie"
    assert parsed.branch == "main"
    assert parsed.path == "universal_robots_ur5e"
