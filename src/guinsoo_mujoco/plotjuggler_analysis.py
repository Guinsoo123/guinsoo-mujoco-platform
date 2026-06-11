from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from guinsoo_mujoco.assets import repo_root
from guinsoo_mujoco.data import RunArtifact
from guinsoo_mujoco.run_catalog import ensure_plotjuggler_csv


class PlotJugglerNotFoundError(FileNotFoundError):
    pass


_QPOS_COLOR = "#1f77b4"
_TARGET_COLOR = "#ff7f0e"
_CTRL_COLOR = "#2ca02c"
_FORCE_COLOR = "#d62728"
_SINGLE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)


def find_plotjuggler_executable() -> Path:
    candidates = [
        Path("/Applications/PlotJuggler.app/Contents/MacOS/PlotJuggler"),
        Path.home() / "Applications/PlotJuggler.app/Contents/MacOS/PlotJuggler",
        Path("/opt/homebrew/bin/plotjuggler"),
        Path("/usr/local/bin/plotjuggler"),
    ]
    found = shutil.which("plotjuggler")
    if found:
        candidates.insert(0, Path(found))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise PlotJugglerNotFoundError(
        "未找到 PlotJuggler。请从 https://github.com/facontidavide/PlotJuggler/releases 安装，"
        "或将其放入 /Applications/PlotJuggler.app。"
    )


def launch_motion_control_analysis(
    artifact: RunArtifact,
    *,
    layout_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Open PlotJuggler with CSV data and a standard motion-control layout."""
    csv_path = ensure_plotjuggler_csv(artifact)
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    joint_names = _joint_names(metadata)
    if layout_path is None:
        layout_path = artifact.hdf5_path.with_name(
            f"{artifact.hdf5_path.stem}_analysis.plotlayout.xml"
        )
    layout_path = Path(layout_path)
    layout_path.write_text(
        build_motion_control_layout(
            joint_names,
            csv_path=csv_path,
            layout_path=layout_path,
        ),
        encoding="utf-8",
    )
    executable = find_plotjuggler_executable()
    subprocess.Popen(
        [
            str(executable),
            "-n",
            "-l",
            str(layout_path),
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return csv_path, layout_path


def build_motion_control_layout(
    joint_names: list[str],
    *,
    csv_path: str | Path | None = None,
    layout_path: str | Path | None = None,
) -> str:
    if not joint_names:
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]

    tabs = [
        _joint_grid_tab(
            "1_关节位置跟踪",
            "t01_position",
            joint_names,
            _paired_signal_curves("qpos", "target", (_QPOS_COLOR, _TARGET_COLOR)),
        ),
        _joint_grid_tab(
            "2_跟踪误差",
            "t02_error",
            joint_names,
            _single_signal_curves("tracking_error"),
        ),
        _joint_grid_tab(
            "3_控制力矩",
            "t03_torque",
            joint_names,
            _paired_signal_curves("ctrl", "actuator_force", (_CTRL_COLOR, _FORCE_COLOR)),
        ),
        _joint_grid_tab(
            "4_关节速度",
            "t04_velocity",
            joint_names,
            _single_signal_curves("qvel"),
        ),
        _joint_grid_tab(
            "5_关节力矩反馈",
            "t05_qfrc",
            joint_names,
            _single_signal_curves("qfrc_actuator"),
        ),
    ]
    tabs_xml = "\n".join(tabs)
    data_source_xml = ""
    if csv_path is not None:
        data_source_xml = _embedded_csv_data_source(
            csv_path=Path(csv_path),
            layout_path=Path(layout_path) if layout_path is not None else None,
        )
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<root>\n"
        f"{data_source_xml}"
        ' <tabbed_widget parent="main_window" name="Main Window">\n'
        f"{tabs_xml}\n"
        " </tabbed_widget>\n"
        "</root>\n"
    )


def _embedded_csv_data_source(
    *,
    csv_path: Path,
    layout_path: Path | None,
) -> str:
    """Embed a single CSV load config so PlotJuggler uses `timestamp` as time axis."""
    if layout_path is not None:
        try:
            csv_ref = csv_path.relative_to(layout_path.parent).as_posix()
        except ValueError:
            csv_ref = csv_path.as_posix()
    else:
        csv_ref = csv_path.name
    return (
        " <previouslyLoaded_Datafiles>\n"
        f'  <fileInfo filename="{csv_ref}" prefix="">\n'
        '   <plugin ID="DataLoad CSV">\n'
        '    <parameters delimiter="0" skip_rows="0" time_axis="timestamp"/>\n'
        "   </plugin>\n"
        "  </fileInfo>\n"
        " </previouslyLoaded_Datafiles>\n"
    )


def bundled_layout_template_path() -> Path:
    return repo_root() / "assets" / "analysis" / "motion_control.plotlayout.xml"


def _joint_names(metadata: dict) -> list[str]:
    names = metadata.get("actuator_names") or metadata.get("joint_names") or []
    return [str(name) for name in names]


def _safe_joint_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def _signal_path(joint: str, suffix: str) -> str:
    return f"joint/{_safe_joint_name(joint)}/{suffix}"


def _plot_xml(curves: list[tuple[str, str]]) -> str:
    curve_xml = "\n".join(
        f'       <curve color="{color}" name="{name}"/>' for name, color in curves
    )
    return (
        '      <plot mode="TimeSeries" flip_x="false" flip_y="false" style="Lines">\n'
        '       <range right="1.000000" top="1.000000" left="0.000000" bottom="0.000000"/>\n'
        '       <limitY max="10" min="-10"/>\n'
        f"{curve_xml}\n"
        "      </plot>"
    )


def _dock_area(plot_xml: str, panel_id: str, joint: str) -> str:
    label = f"{panel_id}__{_safe_joint_name(joint)}"
    return f'     <DockArea name="{label}">\n{plot_xml}\n     </DockArea>'


def _split_sizes(count: int) -> str:
    if count <= 0:
        return "1"
    part = 1.0 / count
    return ";".join(f"{part:.6f}" for _ in range(count))


def _dock_splitter(orientation: str, children: list[str]) -> str:
    count = len(children)
    body = "\n".join(children)
    return (
        f'    <DockSplitter orientation="{orientation}" count="{count}" sizes="{_split_sizes(count)}">\n'
        f"{body}\n"
        "    </DockSplitter>"
    )


def _grid_layout(dock_areas: list[str]) -> str:
    if len(dock_areas) <= 3:
        return _dock_splitter("|", dock_areas)
    rows = [dock_areas[index : index + 3] for index in range(0, len(dock_areas), 3)]
    row_splitters = [_dock_splitter("|", row) for row in rows]
    return _dock_splitter("-", row_splitters)


def _single_signal_curves(suffix: str):
    def builder(joint: str, index: int) -> list[tuple[str, str]]:
        return [
            (
                _signal_path(joint, suffix),
                _SINGLE_COLORS[index % len(_SINGLE_COLORS)],
            )
        ]

    return builder


def _paired_signal_curves(left_suffix: str, right_suffix: str, colors: tuple[str, str]):
    left_color, right_color = colors

    def builder(joint: str) -> list[tuple[str, str]]:
        return [
            (_signal_path(joint, left_suffix), left_color),
            (_signal_path(joint, right_suffix), right_color),
        ]

    return builder


def _joint_grid_tab(
    tab_name: str,
    panel_id: str,
    joint_names: list[str],
    curves_for_joint,
) -> str:
    dock_areas: list[str] = []
    for index, joint in enumerate(joint_names):
        try:
            curves = curves_for_joint(joint, index)
        except TypeError:
            curves = curves_for_joint(joint)
        dock_areas.append(_dock_area(_plot_xml(curves), panel_id, joint))
    grid = _grid_layout(dock_areas)
    return (
        f'  <Tab tab_name="{tab_name}" containers="1">\n'
        "   <Container numdock=\"1\">\n"
        f"{grid}\n"
        "   </Container>\n"
        "  </Tab>"
    )
