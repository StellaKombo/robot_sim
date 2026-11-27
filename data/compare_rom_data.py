"""Plot, replay and export saved ROM comparisons on one simulated-time clock.

No controller or physics is run. At --speed 1 (the default), 60 simulated
seconds play in 60 wall-clock seconds, independently of MPC solve times.
--video exports the dark, side-by-side full-body MuJoCo layout at 1280x720.
--pairs-json accepts reference -> {kcf, single_integrator} (or {kcf, unicycle}) and
exports its clips plus one chaptered MP4. Video export requires FFmpeg.
"""

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The parent folder is a script collection, not a package: KCF_MPC/data.py
# already owns the top-level `data` import in training code. Do not shadow it.
PLOT_DIR = str(Path(__file__).resolve().parent)
if PLOT_DIR not in sys.path:
    sys.path.insert(0, PLOT_DIR)

from plot_rom_data import Q_ROM_LABELS, V_ROM_LABELS, pose_sincos, triangle_vertices
from utils.kcf_reference import PlanarReference

COLORS = ("#0b30c6", "#c30d0d")
VIDEO_SIZE = (1280, 720)
VIDEO_COLORS = ((35, 106, 225), (220, 66, 60))


@dataclass(frozen=True)
class ROMRun:
    label: str
    times: np.ndarray
    pose: np.ndarray  # [world x, world y, sin(theta), cos(theta)]
    velocity: np.ndarray  # body [vx, vy, wz]
    command: np.ndarray  # body [vx, vy, wz]
    settings: dict | None = None
    provenance: dict | None = None
    directory: Path | None = None
    env_index: int = 0
    recorded_reference: tuple | None = None  # time, pose sine/cosine, body twist
    maze: dict | None = None

    def reference(self, times):
        if self.recorded_reference is not None:
            clock, pose, velocity = self.recorded_reference
            times = np.asarray(times, dtype=float)
            if not np.isfinite(times).all():
                raise ValueError('Reference query times must be finite')
            def interpolate(values):
                return np.stack([np.interp(times.ravel(), clock, column)
                                 for column in values.T], axis=-1).reshape(times.shape + (values.shape[1],))
            q, v = interpolate(pose), interpolate(velocity)
            norm = np.linalg.norm(q[..., 2:4], axis=-1, keepdims=True)
            if np.any(norm < 1e-12):
                raise ValueError('Recorded reference has ambiguous interpolated heading')
            q[..., 2:4] /= norm
            return q, v
        if self.settings is None:
            return None, None
        if self.settings.get('reference_mode') == 'maze':
            raise ValueError('Maze replay requires reference_log.npz from measured-progress execution')
        path = PlanarReference(self.settings)
        q, v = path.reference(times)
        tracking = path.track(q, v)
        return tracking[..., :4], tracking[..., 4:7]


def load_rom_run(run, *, env_index=0, label=None):
    """Accept a run name, run directory or NPZ; never guess current references."""
    path = Path(run).expanduser()
    if not path.exists():
        path = ROOT / "data/data" / path
    if path.is_dir():
        path = path / "g1_rom_data.npz"
    with np.load(path, allow_pickle=False) as data:
        pose = pose_sincos(data["q_rom_log"])
        if pose.ndim != 3 or pose.shape[1] < 2:
            raise ValueError(f"{path}: expected at least two samples in (B,T,3/4) poses")
        if not 0 <= env_index < pose.shape[0]:
            raise ValueError(f"{path}: env index must be in [0, {pose.shape[0] - 1}]")
        velocity, command = data["v_rom_log"], data["cmd_log"]
        expected = (*pose.shape[:2], 3)
        if velocity.shape != expected or command.shape != expected:
            raise ValueError(f"{path}: body velocity and command must have shape {expected}")
        if not np.isfinite(velocity).all() or not np.isfinite(command).all():
            raise ValueError(f"{path}: velocity and command must be finite")
        log_dt = float(data["log_dt"] if "log_dt" in data else data["sim_dt"])
        if not np.isfinite(log_dt) or log_dt <= 0:
            raise ValueError(f"{path}: log_dt must be finite and positive")
        times = np.arange(pose.shape[1]) * log_dt
        pose, velocity, command = (array[env_index].copy() for array in (pose, velocity, command))
    plan = path.parent / "trial_plan.json"
    settings = json.loads(plan.read_text())["mpc"] if plan.is_file() else None
    launcher = path.parent / "launcher_result.json"
    provenance = json.loads(launcher.read_text()) if launcher.is_file() else None
    model = settings.get("prediction_model") if settings else None
    recorded = None
    reference_path = path.parent / 'reference_log.npz'
    if reference_path.is_file():
        with np.load(reference_path, allow_pickle=False) as saved:
            clock, reference_pose, reference_velocity = (saved[key].copy() for key in ('time', 'pose', 'velocity'))
        if (clock.ndim != 1 or len(clock) < 2 or reference_pose.shape != (len(clock), 4)
                or reference_velocity.shape != (len(clock), 3)
                or not all(np.isfinite(value).all() for value in (clock, reference_pose, reference_velocity))
                or np.any(np.diff(clock) <= 0.)
                or np.any(np.linalg.norm(reference_pose[:, 2:4], axis=1) < 1e-12)
                or clock[0] > times[0] + 1e-9 or clock[-1] < times[-1] - 1e-9):
            raise ValueError(f'{reference_path}: expected finite increasing times covering the ROM log, '
                             'pose (T,4) and body velocity (T,3)')
        recorded = clock, reference_pose, reference_velocity
    maze_path = path.parent / 'maze.json'
    maze = json.loads(maze_path.read_text()) if maze_path.is_file() else None
    default_label = {"kcf": "KCF", "single_integrator": "Single integrator",
                     "unicycle": "Unicycle"}.get(model, path.parent.name)
    return ROMRun(label or default_label, times, pose, velocity, command, settings, provenance,
                  path.parent.resolve(), env_index, recorded, maze)


def _shared_maze(runs):
    maze = runs[0].maze
    if any(run.maze != maze for run in runs[1:]):
        raise ValueError('Maze comparison requires the same saved maze.json for both runs')
    if maze is None and any((run.settings or {}).get('reference_mode') == 'maze' for run in runs):
        raise ValueError('Maze comparison requires a recorded maze.json')
    return maze


def _maze_polygons(maze):
    polygons = []
    for wall in (maze or {}).get('walls', []):
        angle = wall.get('yaw', 0.)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        polygons.append(np.array(wall['center_xy']) + (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
                        * np.array(wall['size_xy']) / 2.) @ rotation.T)
    return polygons


def comparison_window(runs, start=0., end=None):
    """Use physical simulation time, not sample indices or solver wall time."""
    if len(runs) != 2:
        raise ValueError("ROM comparison requires exactly two runs")
    shared_end = min(run.times[-1] for run in runs)
    end = shared_end if end is None else end
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= shared_end + 1e-9:
        raise ValueError(f"Require 0 <= start < end <= shared logged duration ({shared_end:g} s)")
    return float(start), float(min(end, shared_end))


def sample_pose(run, sim_time):
    """Interpolate sine/cosine, avoiding yaw-wrap interpolation through zero yaw."""
    pose = np.array([np.interp(sim_time, run.times, run.pose[:, i]) for i in range(4)])
    if np.hypot(*pose[2:4]) < 1e-12:
        # An exactly antipodal pair has no unique interpolated heading.
        index = np.clip(np.searchsorted(run.times, sim_time), 0, len(run.times) - 1)
        pose[2:4] = run.pose[index, 2:4]
    return pose


@dataclass(frozen=True)
class _BodyLog:
    q: np.ndarray
    v: np.ndarray
    dt: float
    command_times: np.ndarray
    commands: np.ndarray

    def state(self, sim_time):
        step = np.clip(sim_time / self.dt, 0., len(self.q) - 1)
        first = int(step)
        second = min(first + 1, len(self.q) - 1)
        weight = step - first
        q = (1. - weight) * self.q[first] + weight * self.q[second]
        q[3:7] /= np.linalg.norm(q[3:7])
        return q, self.v[first]

    def command(self, sim_time):
        # Interpolation would invent commands between the trained mode bands.
        index = np.clip(np.searchsorted(self.command_times, sim_time, side="right") - 1,
                        0, len(self.commands) - 1)
        return self.commands[index]


def _load_body_log(run):
    if run.directory is None:
        raise ValueError("MP4 export requires the source run directory")
    path = run.directory / "g1_29dof_data.npz"
    with np.load(path, allow_pickle=False) as saved:
        q, v = saved["q_log"], saved["v_log"]
        if (q.ndim != 3 or v.ndim != 3 or q.shape[:2] != v.shape[:2]
                or q.shape[1] < 2 or q.shape[2] < 7 or not 0 <= run.env_index < len(q)):
            raise ValueError(f"{path}: invalid full-body q/v log shapes")
        q, v = q[run.env_index].astype(float), v[run.env_index].astype(float)
        dt = float(saved["log_dt"] if "log_dt" in saved else saved["sim_dt"])
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError(f"{path}: states and positive log period must be finite")
    norms = np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError(f"{path}: base quaternions must be nonzero")
    q[:, 3:7] /= norms
    # q and -q represent the same orientation: interpolate on one hemisphere.
    flips = np.where(np.sum(q[1:, 3:7] * q[:-1, 3:7], axis=1) < 0., -1., 1.)
    q[1:, 3:7] *= np.cumprod(flips)[:, None]
    command_times, commands = run.times, run.command
    applied = run.directory / "applied_policy_commands.npz"
    if applied.is_file():
        with np.load(applied, allow_pickle=False) as saved:
            command_times, commands = saved["time"].copy(), saved["command"].copy()
        # Preserve the recorded initial command if application starts after t=0.
        if len(command_times) and command_times[0] > run.times[0]:
            command_times = np.r_[run.times[0], command_times]
            commands = np.vstack((run.command[0], commands))
    if (command_times.ndim != 1 or not len(command_times)
            or commands.shape != (len(command_times), 3)
            or not np.isfinite(command_times).all() or not np.isfinite(commands).all()
            or np.any(np.diff(command_times) <= 0.)):
        raise ValueError(f"{applied}: command times must increase and commands must be finite triples")
    return _BodyLog(q, v, dt, command_times, commands)


def _video_times(start, end, fps, speed):
    if (isinstance(fps, bool) or not isinstance(fps, int) or fps < 1
            or not np.isfinite([start, end, speed]).all() or speed <= 0 or start >= end):
        raise ValueError("Video needs a positive integer fps, positive speed and start < end")
    count = max(1, math.ceil((end - start) * fps / speed - 1e-9))
    return start + np.arange(count) * speed / fps


def _tracking_errors(pose, reference):
    distance = np.linalg.norm(pose[..., :2] - reference[..., :2], axis=-1)
    heading = np.rad2deg(np.arctan2(
        pose[..., 2] * reference[..., 3] - pose[..., 3] * reference[..., 2],
        pose[..., 3] * reference[..., 3] + pose[..., 2] * reference[..., 2]))
    return distance * 100., heading


def _write_mp4(output, frames, fps):
    """Stream RGB frames to FFmpeg; replace the destination only on success."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("MP4 export requires ffmpeg on PATH")
    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video output must have an .mp4 extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as tmp:
        temporary = Path(tmp.name)
    process = None
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for frame in frames:
                    if frame.shape != (720, 1280, 3) or frame.dtype != np.uint8:
                        raise ValueError("Video frames must be uint8 RGB arrays with shape (720,1280,3)")
                    process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass  # Report the encoder's stderr below.
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            if process.wait():
                errors.seek(0)
                raise RuntimeError("FFmpeg export failed: " + errors.read().decode(errors="replace"))
        temporary.replace(output)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            if hasattr(frames, "close"):
                frames.close()
        finally:
            temporary.unlink(missing_ok=True)


def _reference_title(mode):
    return {"lissajous": "Figure-eight", "fixed": "Fixed hold",
            "turn180": "180° turn + hold", "stop_turn_go": "Stop-turn-go"}.get(
                mode, mode.replace("_", " ").capitalize())


def _mujoco_video_frames(runs, bodies, model, times, start, end, speed, title, metrics):
    """The original running_unicycle_validation layout, drawn from logged states."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.font_manager import FontProperties, findfont
    from sim.single.simulation import add_arrow, add_sphere

    fonts = {size: ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), size)
             for size in (13, 16, 17, 18, 25)}
    bold = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold")), 26)
    background, light = (16, 20, 28), (224, 228, 235)
    path_times = np.linspace(start, end, max(2, math.ceil((end - start) / .15) + 1))
    paths = [run.reference(path_times)[0][:, :2] for run in runs]
    maze = _shared_maze(runs)
    walls = _maze_polygons(maze)
    selected = [(run.times >= start) & (run.times <= end + 1e-9) for run in runs]
    all_xy = np.concatenate(paths + [run.pose[mask, :2] for run, mask in zip(runs, selected)] + walls)
    lower, upper = all_xy.min(axis=0) - .2, all_xy.max(axis=0) + .2
    scale = min(196 / (upper[0] - lower[0]), 102 / (upper[1] - lower[1]))
    center = (lower + upper) / 2

    def map_xy(xy, offset):
        points = (np.asarray(xy) - center) * [scale, -scale] + [offset + 104, 633]
        return [tuple(point) for point in points.reshape(-1, 2)]

    def add_path(scene, xy, height, width, rgba):
        for a, b in zip(xy[:-1], xy[1:]):
            if np.linalg.norm(a - b) < 1e-9 or scene.ngeom >= scene.maxgeom:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                 np.r_[a, height], np.r_[b, height])
            scene.ngeom += 1

    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.85, 135, -18
    if maze is not None:
        camera.distance, camera.azimuth, camera.elevation = 2.4 * model.stat.extent, 90, -65
    renderer = mujoco.Renderer(model, height=480, width=640, max_geom=5000)
    try:
        for sim_time in times:
            references = [run.reference(sim_time)[0] for run in runs]
            # A common world camera preserves measured offsets between the runs.
            camera.lookat[:] = (model.stat.center if maze is not None else
                                [*np.mean([ref[:2] for ref in references], axis=0), .72])
            canvas = Image.new("RGB", VIDEO_SIZE, background)
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 8), f"{title}  |  recorded simulation  |  t={sim_time:.2f} / {end:.2f} s  |  {speed:g}x",
                      font=fonts[25], fill=light)
            for panel, (run, body, reference, path, metric, color) in enumerate(
                    zip(runs, bodies, references, paths, metrics, VIDEO_COLORS)):
                offset = panel * 640
                draw.text((offset + 16, 44), run.label, font=bold, fill=color)
                controls = ("vx and wz controls, vy=0" if np.all(body.commands[:, 1] == 0.)
                            else "vx, vy and wz controls")
                label_right = draw.textbbox((offset + 16, 44), run.label, font=bold)[2]
                draw.text((max(offset + 200, label_right + 16), 51), f"{model.nu}-DoF G1  |  {controls}",
                          font=fonts[16], fill=(193, 198, 208))
                q, v = body.state(sim_time)
                data.qpos[:], data.qvel[:], data.time = q, v, sim_time
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                add_path(renderer.scene, path, .018, .006, (.45, .48, .52, .6))
                first = np.searchsorted(run.times, start)
                last = np.searchsorted(run.times, sim_time, side="right")
                pose = sample_pose(run, sim_time)
                trail = np.vstack((run.pose[first:last:10, :2], pose[:2]))
                add_path(renderer.scene, trail, .028, .008, (*np.asarray(color) / 255., .9))
                target = [*reference[:2], .045]
                add_sphere(renderer.scene, target, .035, [1., .55, .05, 1.])
                add_arrow(renderer.scene, target,
                          [reference[0] + .32 * reference[3], reference[1] + .32 * reference[2], .045],
                          [1., .55, .05, 1.], width=.012)
                canvas.paste(Image.fromarray(renderer.render()), (offset, 80))
                draw = ImageDraw.Draw(canvas)
                distance, heading = _tracking_errors(pose, reference)
                command = body.command(sim_time)
                draw.text((offset + 14, 567),
                          f"RMSE: {metric['position_rmse_cm']:.2f} cm position  |  {metric['heading_rmse_deg']:.2f} deg heading",
                          font=fonts[17], fill=light)
                draw.text((offset + 14, 594), f"Current error: {distance:.1f} cm  |  {heading:+.1f} deg",
                          font=fonts[18], fill=light)
                draw.text((offset + 14, 621),
                          f"vx,vy=[{command[0]:+.2f},{command[1]:+.2f}] m/s; wz={command[2]:+.2f} rad/s",
                          font=fonts[16], fill=(193, 198, 208))
                result = (run.provenance or {}).get("simulator", {})
                outcome = result.get("outcome", "not recorded")
                draw.text((offset + 14, 648), f"Full trial outcome: {outcome}",
                          font=fonts[16], fill=(193, 198, 208))
                if 'contact_episodes' in result:
                    episodes = result['contact_episodes']
                    count = sum(episode['start_s'] <= sim_time for episode in episodes)
                    active = any(episode['start_s'] <= sim_time <= episode['end_s'] for episode in episodes)
                    draw.text((offset + 14, 674),
                              f"Wall contact episodes: {count}" + ('  |  CONTACT EPISODE' if active else ''),
                              font=fonts[16], fill=(255, 150, 20) if active else light)
                mini_x = offset + 418
                draw.rectangle((mini_x, 575, mini_x + 208, 691),
                               fill=(33, 40, 52), outline=(93, 102, 117))
                for wall in walls:
                    draw.polygon(map_xy(wall, mini_x), fill=(93, 102, 117))
                draw.line(map_xy(path, mini_x), fill=(165, 172, 184), width=1)
                if len(trail) > 1:
                    draw.line(map_xy(trail, mini_x), fill=color, width=2)
                for point, marker in [(reference[:2], (255, 150, 20)), (pose[:2], color)]:
                    x, y = map_xy(point, mini_x)[0]
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=marker)
            draw.line((640, 40, 640, 696), fill=(65, 74, 88), width=2)
            draw.text((12, 699),
                      f"Gray: reference path. Orange: target. Colored: actual trail. RMSE uses shared {start:g}–{end:g}s. No new physics/MPC.",
                      font=fonts[13], fill=(216, 214, 187))
            yield np.asarray(canvas)
    finally:
        renderer.close()


def export_rom_comparison_video(runs, output, *, start=0., end=None, speed=1., fps=30, title=None):
    """Replay full-body logs as an H.264 MP4, without policy inference or mj_step.

    Defaults match running_unicycle_validation: dark 1280x720 side-by-side
    panels, 30 fps, reference/trail overlays, errors, commands and minimaps.
    MUJOCO_GL may select another backend; Linux export defaults to EGL.
    """
    start, end = comparison_window(runs, start, end)
    times = _video_times(start, end, fps, speed)
    if shutil.which("ffmpeg") is None:
        raise ValueError("MP4 export requires ffmpeg on PATH")
    bodies, metrics = [], []
    for run in runs:
        if run.settings is None:
            raise ValueError("MP4 export requires each run's trial_plan.json saved reference")
        body = _load_body_log(run)
        if (len(body.q) - 1) * body.dt < end - 1e-9:
            raise ValueError(f"{run.label}: full-body log does not cover the selected interval")
        bodies.append(body)
        mask = (run.times >= start) & (run.times <= end + 1e-9)
        if not mask.any():
            raise ValueError("Selected interval must contain a recorded ROM sample from each run")
        reference, _ = run.reference(run.times[mask])
        distance, heading = _tracking_errors(run.pose[mask], reference)
        metrics.append(dict(position_rmse_cm=float(np.sqrt(np.mean(distance**2))),
                            heading_rmse_deg=float(np.sqrt(np.mean(heading**2)))))
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from sim.single.simulation import load_model, XML_PATH
    maze = _shared_maze(runs)
    model = load_model(XML_PATH) if maze is None else load_model(XML_PATH, maze=maze)
    if any(body.q.shape[1] != model.nq or body.v.shape[1] != model.nv for body in bodies):
        raise ValueError(f"Full-body logs must match the replay model: nq={model.nq}, nv={model.nv}")
    title = title or _reference_title(runs[0].settings["reference_mode"])
    _write_mp4(output, _mujoco_video_frames(runs, bodies, model, times, start, end, speed, title, metrics), fps)
    result = dict(file=str(Path(output).resolve()), title=title, fps=fps, speed=speed,
                  frames=len(times), duration_s=len(times) / fps, width=1280, height=720,
                  shared_interval_s=[start, end], metrics=dict(zip(("left", "right"), metrics)),
                  runs=[str(run.directory) for run in runs], full_body_recorded_states=True)
    print(f"Saved video to {output} ({len(times) / fps:.2f} s, {fps} fps)", flush=True)
    return result


def replay_time(wall_start, wall_now, start, end, speed=1.):
    """Late renders jump to the current instant instead of slowing the recording."""
    return min(end, start + max(0., wall_now - wall_start) * speed)


def warn_if_unmatched(runs):
    a, b = runs
    _shared_maze(runs)
    for run in runs:
        if run.provenance and (not run.provenance.get("sources_unchanged", False)
                               or run.provenance.get("interrupted", False)
                               or run.provenance.get("simulator", {}).get("outcome") != "complete"):
            warnings.warn(f"{run.label}: launcher flags changed sources, interruption or an incomplete run; "
                          "replay is available but this is not a verified clean benchmark.", stacklevel=2)
    if not np.allclose(a.pose[0], b.pose[0], atol=1e-6, rtol=0):
        warnings.warn("Initial poses differ; playback preserves their recorded world coordinates.", stacklevel=2)
    end = min(a.times[-1], b.times[-1])
    q_a, _ = a.reference(np.linspace(0, end, 101))
    q_b, _ = b.reference(np.linspace(0, end, 101))
    if q_a is None or q_b is None:
        warnings.warn("A trial_plan.json is missing; no current/default reference will be substituted.", stacklevel=2)
    elif not np.allclose(q_a, q_b, atol=1e-6, rtol=0):
        warnings.warn("Saved references differ; each panel shows its own recorded task.", stacklevel=2)


def plot_rom_comparison(runs, *, start=0., end=None):
    """Physical pose sine/cosine and body twist, with saved task/command overlays."""
    start, end = comparison_window(runs, start, end)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for run, color in zip(runs, COLORS):
        selected = (run.times >= start) & (run.times <= end + 1e-9)
        t = run.times[selected]
        q, v, command = run.pose[selected], run.velocity[selected], run.command[selected]
        q_ref, v_ref = run.reference(t)
        for i, ax in enumerate(axes[0]):
            ax.plot(t, q[:, i], color=color, linewidth=1., label=run.label)
            if q_ref is not None:
                ax.plot(t, q_ref[:, i], "--", color=color, alpha=.6,
                        linewidth=.9, label=f"{run.label} reference")
        for i, ax in enumerate(axes[1, :3]):
            ax.plot(t, v[:, i], color=color, linewidth=1.)
            ax.plot(t, command[:, i], ":", color=color, alpha=.65, linewidth=.8)
            if v_ref is not None:
                ax.plot(t, v_ref[:, i], "--", color=color, alpha=.6, linewidth=.9)
        if q_ref is not None:
            axes[1, 3].plot(t, np.linalg.norm(q[:, :2] - q_ref[:, :2], axis=1), color=color)
    for ax, title in zip(axes.flat, Q_ROM_LABELS + V_ROM_LABELS + ["Position error [m]"]):
        ax.set(title=title, xlabel="Simulated time [s]", xlim=(start, end))
        ax.grid(True, alpha=.25)
    axes[0, 2].set_ylim(-1.1, 1.1)
    axes[0, 3].set_ylim(-1.1, 1.1)
    axes[0, 0].legend(fontsize=8)
    axes[1, 3].text(.03, .96, "Solid: measured\nDashed: saved reference\nDotted: applied command (body twist)",
                    transform=axes[1, 3].transAxes, va="top", fontsize=8)
    fig.suptitle("ROM comparison — model-aligned heading features; world pose / body twist")
    return fig


class ROMComparisonPlayback:
    """Reusable artists; update(t) renders both recordings at the same sim time."""

    def __init__(self, runs, *, start=0., end=None, overlay=True, speed=1.):
        self.start, self.end = comparison_window(runs, start, end)
        self.runs, self.speed = runs, speed
        count = 1 if overlay else len(runs)
        self.fig, axes = plt.subplots(1, count, figsize=(7 * count, 6), squeeze=False,
                                      constrained_layout=True)
        self.axes = axes[0]
        self.triangles, self.trails, self.references = [], [], []
        all_positions = [run.pose[:, :2] for run in runs]
        walls = _maze_polygons(_shared_maze(runs))
        all_positions.extend(walls)
        for run in runs:
            q_ref, _ = run.reference(run.times)
            if q_ref is not None:
                all_positions.append(q_ref[:, :2])
        xy = np.concatenate(all_positions)
        lower, upper = xy.min(axis=0) - .5, xy.max(axis=0) + .5
        for ax in self.axes:
            for wall in walls:
                ax.add_patch(Polygon(wall, facecolor='#687686', edgecolor='none', zorder=0))
            ax.set(xlabel="World x [m]", ylabel="World y [m]",
                   xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=.25)
        for i, (run, color) in enumerate(zip(runs, COLORS)):
            ax = self.axes[0 if overlay else i]
            q_ref, _ = run.reference(run.times)
            reference_marker = None
            if q_ref is not None:
                selected = (run.times >= self.start) & (run.times <= self.end + 1e-9)
                ax.plot(q_ref[selected, 0], q_ref[selected, 1], "--", color="gray", linewidth=1.)
                reference_marker = Polygon(triangle_vertices(q_ref[0], .18, .13),
                                           facecolor="none", edgecolor="gray", linewidth=1.5,
                                           label="Reference" if not overlay or i == 0 else None)
                ax.add_patch(reference_marker)
            triangle = Polygon(triangle_vertices(run.pose[0]), facecolor=color,
                               edgecolor="black", alpha=.85, label=run.label, zorder=3)
            ax.add_patch(triangle)
            trail, = ax.plot([], [], color=color, linewidth=1.1, alpha=.8)
            self.triangles.append(triangle)
            self.trails.append(trail)
            self.references.append(reference_marker)
            if not overlay:
                ax.set_title(run.label)
        for ax in self.axes:
            ax.legend(loc="best", fontsize=8)
        self.title = self.fig.suptitle("")
        self.update(self.start)

    def update(self, sim_time):
        sim_time = float(np.clip(sim_time, self.start, self.end))
        for i, run in enumerate(self.runs):
            pose = sample_pose(run, sim_time)
            self.triangles[i].set_xy(triangle_vertices(pose))
            first = np.searchsorted(run.times, self.start, side="left")
            last = np.searchsorted(run.times, sim_time, side="right")
            trail = np.vstack((run.pose[first:last, :2], pose[:2]))
            self.trails[i].set_data(trail[:, 0], trail[:, 1])
            if self.references[i] is not None:
                q_ref, _ = run.reference(sim_time)
                self.references[i].set_xy(triangle_vertices(q_ref, .18, .13))
        self.title.set_text(f"ROM comparison — t={sim_time:.2f} / {self.end:.2f} s — {self.speed:g}× playback")


def playback_rom_comparison(runs, *, start=0., end=None, speed=1., render_hz=50.,
                            loop=True, overlay=True):
    """Wall-clock-paced saved-data replay; slow drawing skips obsolete frames."""
    if not np.isfinite([speed, render_hz]).all() or speed <= 0 or render_hz <= 0:
        raise ValueError("speed and render_hz must be finite and positive")
    playback = ROMComparisonPlayback(runs, start=start, end=end, overlay=overlay, speed=speed)
    was_interactive = plt.isinteractive()
    plt.ion()
    plt.show(block=False)
    playback.fig.canvas.draw()
    try:
        while plt.fignum_exists(playback.fig.number):
            wall_start = time.perf_counter()
            while plt.fignum_exists(playback.fig.number):
                frame_start = time.perf_counter()
                current = replay_time(wall_start, frame_start, playback.start, playback.end, speed)
                playback.update(current)
                playback.fig.canvas.draw_idle()
                playback.fig.canvas.flush_events()
                if current >= playback.end:
                    break
                # Pause pumps GUI events; rendering time counts towards playback time.
                slack = 1. / render_hz - (time.perf_counter() - frame_start)
                plt.pause(max(.001, min(max(.001, slack), (playback.end - current) / speed)))
            if not loop:
                break
    finally:
        if not was_interactive:
            plt.ioff()
    return playback


def _combine_videos(clips, output):
    """Concatenate this export's clips and preserve one chapter per reference."""
    output = Path(output).resolve()
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, suffix=".txt", delete=False) as listing:
            temporary.append(Path(listing.name))
            for clip in clips:
                # Batch filenames below contain only validated reference names.
                listing.write(f"file 'videos/{Path(clip['file']).name}'\n")
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, suffix=".txt", delete=False) as metadata:
            temporary.append(Path(metadata.name))
            metadata.write(";FFMETADATA1\n")
            elapsed = 0.
            for clip in clips:
                stop = elapsed + clip["duration_s"]
                metadata.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(elapsed * 1000)}\n"
                               f"END={round(stop * 1000)}\ntitle={clip['title']}\n")
                elapsed = stop
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as encoded:
            temporary.append(Path(encoded.name))
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "1", "-i", listing.name, "-i", metadata.name,
                        "-map_metadata", "1", "-map_chapters", "1", "-c", "copy",
                        "-movflags", "+faststart", encoded.name], check=True, capture_output=True)
        Path(encoded.name).replace(output)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _export_pairs(args):
    if args.run_a or args.run_b or not args.video:
        raise ValueError("--pairs-json requires --video OUTPUT.mp4 instead of positional runs")
    pairs = json.loads(args.pairs_json.read_text())
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("Pair manifest must map reference names to {kcf: run, single_integrator/unicycle: run}")
    # Validate manifest names and schema before generating any output.
    for reference, pair in pairs.items():
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", reference)
                or not isinstance(pair, dict)
                or set(pair) not in ({"kcf", "single_integrator"}, {"kcf", "unicycle"})
                or not all(isinstance(value, str) for value in pair.values())):
            raise ValueError(f"Invalid reference/run pair: {reference!r}")
    clips = []
    for reference, pair in pairs.items():
        baseline = 'single_integrator' if 'single_integrator' in pair else 'unicycle'
        models = ('kcf', baseline)
        sources = [args.pairs_json.parent / pair[model] for model in models]
        runs = [load_rom_run(path if path.exists() else pair[model], env_index=args.env_index)
                for path, model in zip(sources, models)]
        for run, model in zip(runs, models):
            if run.settings is not None and run.settings.get('prediction_model') != model:
                raise ValueError(f"{reference}: manifest selects {model}, but {run.directory} records "
                                 f"{run.settings.get('prediction_model')}")
        if args.labels:
            runs = [replace(run, label=label) for run, label in zip(runs, args.labels)]
        warn_if_unmatched(runs)
        if args.output_dir:
            plt.close(_save_comparison_plots(runs, args.output_dir / reference, args))
        output = args.video.parent / "videos" / f"{reference}_kcf_vs_{baseline}.mp4"
        clips.append(export_rom_comparison_video(
            runs, output, start=args.start, end=args.end, fps=args.fps,
            speed=args.speed, title=_reference_title(reference)))
    _combine_videos(clips, args.video)
    manifest = args.video.parent / "video_manifest.json"
    manifest.write_text(json.dumps(dict(clips=clips, combined=str(args.video.resolve()),
                                       duration_s=sum(clip["duration_s"] for clip in clips)), indent=2) + "\n")
    print(f"Saved combined video to {args.video}; source runs and intervals: {manifest}")


def _save_comparison_plots(runs, output, args):
    start, end = comparison_window(runs, args.start, args.end)
    output.mkdir(parents=True, exist_ok=True)
    static = plot_rom_comparison(runs, start=start, end=end)
    static.savefig(output / "rom_states.png", dpi=160)
    preview = ROMComparisonPlayback(runs, start=start, end=end, overlay=args.overlay, speed=args.speed)
    preview.update(end)
    preview.fig.savefig(output / "rom_playback_preview.png", dpi=160)
    plt.close(preview.fig)
    print(f"Saved comparison plots to {output}")
    return static


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", nargs="?", help="First run name, directory or ROM NPZ")
    parser.add_argument("run_b", nargs="?", help="Second run name, directory or ROM NPZ")
    parser.add_argument("--video", type=Path, help="Export the original dark full-body MP4 layout; skips GUI playback")
    parser.add_argument("--fps", type=int, default=30, help="MP4 frames per second (default 30)")
    parser.add_argument("--pairs-json", type=Path,
                        help="Export {kcf, single_integrator} or {kcf, unicycle} pairs and combine into --video")
    parser.add_argument("--title", help="Single-pair video title; defaults to the saved reference mode")
    parser.add_argument("--labels", nargs=2, help="Optional legend labels, in run order")
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1., help="Simulated seconds per playback/video second (default 1)")
    parser.add_argument("--render-hz", type=float, default=50.)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--overlay", dest="overlay", action="store_true",
                        help="Draw both robots on one shared axis (default)")
    layout.add_argument("--side-by-side", dest="overlay", action="store_false",
                        help="Draw each robot on a separate axis")
    parser.set_defaults(overlay=True)
    parser.add_argument("--once", action="store_true", help="Play once rather than looping")
    parser.add_argument("--no-playback", action="store_true", help="Skip interactive playback; usable with MPLBACKEND=Agg")
    parser.add_argument("--output-dir", type=Path, help="Save comparison plots (one subfolder per reference in batch mode)")
    args = parser.parse_args(argv)
    try:
        if not np.isfinite([args.speed, args.render_hz]).all() or min(args.speed, args.render_hz) <= 0:
            raise ValueError("speed and render-hz must be finite and positive")
        if args.fps < 1:
            raise ValueError("Video fps must be positive")
        if args.video and args.video.suffix.lower() != ".mp4":
            raise ValueError("Video output must have an .mp4 extension")
        if args.pairs_json:
            _export_pairs(args)
            plt.close("all")
            return
        if not args.run_a or not args.run_b:
            raise ValueError("Provide two runs, or --pairs-json MANIFEST --video OUTPUT.mp4")
        runs = [load_rom_run(value, env_index=args.env_index) for value in (args.run_a, args.run_b)]
        if args.labels:
            runs = [replace(run, label=label) for run, label in zip(runs, args.labels)]
        elif runs[0].label == runs[1].label:
            runs = [replace(run, label=f"{run.label} ({Path(value).name})")
                    for run, value in zip(runs, (args.run_a, args.run_b))]
        start, end = comparison_window(runs, args.start, args.end)
        warn_if_unmatched(runs)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Comparing {runs[0].label} and {runs[1].label}: {start:g}–{end:g} simulated seconds")
    print(f"Playback: {args.speed:g}×; {(end - start) / args.speed:g} wall seconds per pass. No physics/controller is run.")
    if args.output_dir:
        static = _save_comparison_plots(runs, args.output_dir, args)
    else:
        static = plot_rom_comparison(runs, start=start, end=end)
    if args.video:
        try:
            export_rom_comparison_video(runs, args.video, start=start, end=end, speed=args.speed,
                                        fps=args.fps, title=args.title)
        except (ValueError, OSError, KeyError, ImportError, RuntimeError) as error:
            parser.error(str(error))
        plt.close(static)
        return
    if not args.no_playback:
        if plt.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
            parser.error("Playback needs an interactive Matplotlib backend; use --no-playback for headless export")
        print("Close the playback window to stop. Solid triangles: measured; hollow gray: reference.")
        playback_rom_comparison(runs, start=start, end=end, speed=args.speed,
                                render_hz=args.render_hz, loop=not args.once, overlay=args.overlay)
    if args.no_playback and plt.get_backend().lower() == "agg":
        plt.close(static)
    else:
        plt.show()


if __name__ == "__main__":
    main()
