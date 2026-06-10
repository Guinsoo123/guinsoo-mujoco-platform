from guinsoo_mujoco.cli import main


def test_fetch_assets_reports_unknown_license_without_traceback(tmp_path, capsys):
    code = main(["fetch-assets", "xlerobot", "--cache-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert "许可证为 UNKNOWN" in captured.err
    assert "Traceback" not in captured.err
