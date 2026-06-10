import numpy as np

from guinsoo_mujoco.cli import main
from guinsoo_mujoco.data import RunMetadata, RunRecorder


def test_fetch_assets_reports_unknown_license_without_traceback(tmp_path, capsys):
    code = main(["fetch-assets", "xlerobot", "--cache-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert "许可证为 UNKNOWN" in captured.err
    assert "Traceback" not in captured.err


def test_export_plotjuggler_writes_csv(tmp_path, capsys):
    recorder = RunRecorder(
        tmp_path,
        RunMetadata(
            robot_id="ur5e",
            controller="joint_position",
            demo="joint_position",
            mujoco_version="test",
            asset_source="source",
            app_version="0.1.0",
            config={},
            actuator_names=("j1",),
        ),
    )
    recorder.append(
        t=0.0,
        qpos=np.array([0.0]),
        qvel=np.array([0.0]),
        ctrl=np.array([1.0]),
        target=np.array([0.2]),
        actuator_force=np.array([2.0]),
        qfrc_actuator=np.array([3.0]),
    )
    artifact = recorder.close()

    code = main(["export-plotjuggler", str(artifact.hdf5_path), "--csv", str(tmp_path / "out.csv")])

    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "out.csv").exists()
    assert "PlotJuggler CSV" in captured.out
