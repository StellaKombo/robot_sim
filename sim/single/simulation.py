##
#
#  Simulation
#
##

# standard imports
import argparse
import contextlib
import copy
import math
import os
import sys
import threading
import time

# third-party imports
import numpy as np
import torch
import mujoco
import mujoco.viewer
import zmq

# repo root: ROBOT_SIM_ROOT_DIR from the conda env, else three levels up from this file
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)
from policy.actor import load_policy
from policy.config import G1Config
from utils.math_utils import gravity_from_quat, q_to_rom, v_to_rom
from utils.zmq_utils import CMD_PORT, STATE_PORT, bind_pub, bind_sub, pack_state, unpack_twist

# GLOBAL VARIABLES
XML_PATH = os.path.join(ROOT_DIR, "deploy_robot", "models", "g1", "g1_29dof_mode11_scene.xml")
POLICY_PATH = os.path.join(ROOT_DIR, "deploy_robot", "policy", "g1", "g1_29dof_vel.onnx")
SIM_DT = 0.002      # [s] physics timestep
LOG_HZ = 100.0      # [Hz] default logging rate; must divide the sim rate exactly
RENDER_HZ = 50.0    # [Hz] visual rendering frequency
CMD_TIMEOUT = 0.5   # [s] without a command from a controller before falling back to zero

# world-frame RGB triad at the origin (X=red, Y=green, Z=blue), as in mj-nlp's replay
AXIS_LEN = 0.2        # [m] length of each axis cylinder
AXIS_RADIUS = 0.008   # [m] radius of each axis cylinder
AXIS_ALPHA = 0.3      # opacity of the axis cylinders (0 = invisible, 1 = opaque)

# overlays: the controller's reference point, and commanded vs actual twist arrows over the head
REF_RADIUS = 0.05                     # [m] reference ball radius
REF_RGBA = (1.0, 0.5, 0.0, 0.8)       # orange
ARROW_SCALE = 0.5                     # [m] of arrow per m/s or rad/s
ARROW_HEIGHT = 0.65                   # [m] arrows start this far above the pelvis, over the head
ARROW_WIDTH = 0.015                   # [m]
CMD_RGBA = (0.1, 0.3, 1.0, 0.8)       # commanded vx, vy and wz: blue
ACT_RGBA = (1.0, 0.15, 0.15, 0.8)     # actual vx, vy and wz: red

# timing legend in the viewer's top-left corner, as in deploy_robot's simulation
LEGEND_FONT = getattr(mujoco.mjtFontScale, "mjFONTSCALE_250",
                      getattr(mujoco.mjtFontScale, "mjFONTSCALE_200", mujoco.mjtFontScale.mjFONTSCALE_150))


############################################################################
# MODEL
############################################################################

def load_model(xml_path: str) -> mujoco.MjModel:
    """Load the XML and add the world-frame triad; its geoms are massless and never collide."""
    spec = mujoco.MjSpec.from_file(xml_path)
    body = spec.worldbody.add_body(name="world_axes")
    for fromto, rgba in (
        ([0, 0, 0, AXIS_LEN, 0, 0], [1, 0, 0, AXIS_ALPHA]),   # +X red
        ([0, 0, 0, 0, AXIS_LEN, 0], [0, 1, 0, AXIS_ALPHA]),   # +Y green
        ([0, 0, 0, 0, 0, AXIS_LEN], [0, 0, 1, AXIS_ALPHA]),   # +Z blue
    ):
        g = body.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.fromto = fromto
        g.size = [AXIS_RADIUS, 0, 0]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0
        g.density = 0.0
    return spec.compile()


############################################################################
# OVERLAY GEOMS
############################################################################

def add_sphere(scn: mujoco.MjvScene, center, radius: float, rgba):
    """Append a decorative sphere to a scene; skipped if the scene is full."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(center, dtype=np.float64), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def add_arrow(scn: mujoco.MjvScene, start, end, rgba, width: float = ARROW_WIDTH):
    """Append a decorative arrow from `start` to `end`; skipped if the scene is full."""
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64))
    scn.ngeom += 1


############################################################################
# PACING
############################################################################

class Pacer:
    """Keeps sim time in step with wall time; when behind, runs without sleeping until caught up."""

    def __init__(self):
        self.wall0 = time.perf_counter()

    def start(self, sim_t: float):
        # wall time that corresponds to sim time 0
        self.wall0 = time.perf_counter() - sim_t

    def wait(self, sim_t: float):
        # sleep only while sim time is ahead of wall time
        slack = self.wall0 + sim_t - time.perf_counter()
        if slack > 0:
            time.sleep(slack)


############################################################################
# SIMULATION
############################################################################

class Simulation:
    """G1 in MuJoCo running the velocity policy, driven by twist commands from a control_*.py.

    The policy runs every control_dt in lockstep with physics, and PD runs every physics step.
    Before a controller connects, or after CMD_TIMEOUT seconds without a message, the command
    is zero.
    """

    def __init__(self, xml_path: str, policy_path: str, sim_dt: float, robot_cfg: G1Config = None,
                 headless: bool = False, log_name: str = None, log_hz: float = LOG_HZ,
                 readonly_viewer: bool = False):
        # config and run options
        self.cfg = robot_cfg or G1Config()
        self.nu = len(self.cfg.Kp)
        self.headless = headless
        # Qualification is watch-only: GUI actions must never mutate physics.
        # Normal interactive/manual simulation keeps its existing shared-data viewer.
        self.readonly_viewer = readonly_viewer
        self._display_model = self._display_data = None
        self.log_name = log_name     # run name; the log goes to data/data/<log_name>/

        # model and data
        self.model = load_model(xml_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        
        # pelvis gyro's slot in sensordata, looked up by name
        gyro_adr = self.model.sensor("pelvis_imu_gyro_sensor").adr[0]
        self.gyro_slice = slice(gyro_adr, gyro_adr + 3)
        self.pelvis_id = self.model.body("pelvis").id

        # physics steps per policy query
        ratio = self.cfg.control_dt / sim_dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError(f"sim_dt ({sim_dt}) must divide control_dt ({self.cfg.control_dt}) exactly.")
        self.decimation = int(round(ratio))

        # physics steps per logged sample
        log_ratio = 1.0 / (log_hz * sim_dt) if log_hz > 0 else 0.0
        if log_ratio < 1 or not math.isclose(log_ratio, round(log_ratio), abs_tol=1e-9):
            raise ValueError(f"log_hz ({log_hz}) must divide the sim rate ({1.0 / sim_dt:g} Hz) exactly.")
        self.log_decimation = int(round(log_ratio))
        self.log_dt = self.log_decimation * sim_dt

        # policy
        self.policy = load_policy(policy_path)
        self.cmd_scale = np.asarray(self.cfg.cmd_scale, dtype=np.float32)
        self.default_j = np.asarray(self.cfg.default_joint_pos, dtype=np.float32)
        self.reset()

    def reset(self):
        """Nominal pose, zero velocity, zero command, cleared policy memory and log."""
        
        # robot at the nominal pose, at rest
        self.data.qpos[:7] = self.cfg.default_base_pos
        self.data.qpos[7:7 + self.nu] = self.cfg.default_joint_pos
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        
        # zero command, fresh policy memory, empty log
        self.cmd = np.zeros(3, dtype=np.float32)
        self.ref = None   # controller's reference point [x, y], drawn when set
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.action = self.prev_action.copy()
        self.n_steps = 0   # physics steps since reset, to pace logging
        self.log_q, self.log_v, self.log_cmd = [], [], []
        
        # recompute kinematics and sensors for the new state
        mujoco.mj_forward(self.model, self.data)

    # COMMAND

    def set_cmd(self, cmd: np.ndarray):
        """Body-frame [vx, vy, wz], clipped to the policy's training range."""
        self.cmd = np.clip(np.asarray(cmd, dtype=np.float32), -self.cmd_scale, self.cmd_scale)

    # POLICY

    def build_observation(self) -> np.ndarray:
        """98-dim policy observation, in the exact layout the policy was trained on."""
        
        d = self.data
        
        # sensor readings, cast to float32 as in training
        omega = d.sensordata[self.gyro_slice].astype(np.float32)
        gravity = gravity_from_quat(d.qpos[3:7].astype(np.float32))
        qpos_j = d.qpos[7:7 + self.nu].astype(np.float32)
        qvel_j = d.qvel[6:6 + self.nu].astype(np.float32)

        # gait clock from sim time, zeroed below stand_cmd_threshold
        phase = (d.time % self.cfg.gait_period) / self.cfg.gait_period
        gait_phase = np.array([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
                              dtype=np.float32)
        if np.linalg.norm(self.cmd) < self.cfg.stand_cmd_threshold:
            gait_phase[:] = 0.0

        # assemble in the order the policy was trained on
        obs = np.concatenate([
            omega,                      # 0:3
            gravity,                    # 3:6
            self.cmd,                   # 6:9
            gait_phase,                 # 9:11
            qpos_j - self.default_j,    # 11:40
            qvel_j,                     # 40:69
            self.prev_action,           # 69:98
        ]).astype(np.float32)
        return obs

    def policy_step(self):
        """One policy query; the action is held until the next one."""
        # run the policy; the action also feeds the next observation
        self.action = self.policy.inference(self.build_observation()).astype(np.float32)
        self.prev_action = self.action

    # PHYSICS

    def step(self):
        """One physics step: PD torque toward the held action's joint targets, mj_step, log if due."""
        
        if self.log_name is not None and not self.log_q:
            self._record()   # initial state, logged with the first command
        
        # PD torque toward the joint targets encoded by the held action
        qpos_des = self.action * self.cfg.action_scale + self.cfg.default_joint_pos
        qpos_j = self.data.qpos[7:7 + self.nu]
        qvel_j = self.data.qvel[6:6 + self.nu]
        self.data.ctrl[:] = self.cfg.Kp * (qpos_des - qpos_j) + self.cfg.Kd * (-qvel_j)
        
        # advance physics, then log the new state every log_decimation steps
        mujoco.mj_step(self.model, self.data)
        self.n_steps += 1
        if self.log_name is not None and self.n_steps % self.log_decimation == 0:
            self._record()

    def _record(self):
        # full state plus the command in effect
        self.log_q.append(self.data.qpos.copy())
        self.log_v.append(self.data.qvel.copy())
        self.log_cmd.append(self.cmd.copy())

    # RUN LOOP

    def run(self, duration: float = None):
        """Real time until the viewer closes; each policy tick sends the state, then reads a command."""
        
        # command and state sockets, optional step budget, and real-time pacer
        ctx = zmq.Context.instance()
        cmd_sub = bind_sub(ctx, CMD_PORT)
        state_pub = bind_pub(ctx, STATE_PORT)
        n_steps = None if duration is None else int(round(duration / self.model.opt.timestep))
        pacer = Pacer()
        print(f"Listening for commands on port {CMD_PORT}, publishing state on port {STATE_PORT}; "
              "using zero command until a controller connects.")

        # whether commands are arriving, and when the last one came
        connected = False
        last_rx = None
        warned_bad_msg = False
        try:
            # loop until the duration ends or the viewer closes
            with self._viewer() as viewer:
                pacer.start(self.data.time)
                next_render = time.perf_counter()
                self.wall_start = next_render                    # for the legend's real time
                self.rtr_last = (self.data.time, next_render)    # for the legend's real-time rate
                k = 0
                while (n_steps is None or k < n_steps) and (viewer is None or viewer.is_running()):
        
                    # policy tick: send the state, read the newest command, check timeout, run policy
                    if k % self.decimation == 0:
                        state_pub.send(pack_state(self.data.time, self.data.qpos, self.data.qvel,
                                                  self.prev_action))
                        try:
                            cmd, self.ref = unpack_twist(cmd_sub.recv(zmq.NOBLOCK))
                            self.set_cmd(cmd)
                            last_rx = time.perf_counter()
                            if not connected:
                                print("Receiving commands from the controller.")
                                connected = True
                        except zmq.Again:
                            pass
                        except ValueError as err:
                            # e.g. a controller from an older version of this code; keep running
                            if not warned_bad_msg:
                                print(f"Ignoring a malformed command ({err}). Restart the controller.")
                                warned_bad_msg = True
        
                        # controller went quiet: fall back to zero
                        if connected and time.perf_counter() - last_rx > CMD_TIMEOUT:
                            print(f"No command for {CMD_TIMEOUT} s; using zero command.")
                            connected = False
                            self.set_cmd(np.zeros(3))
                            self.ref = None
                        self.policy_step()

                    # every physics step: PD and physics, render if due, keep real time
                    self.step()
                    k += 1
                    next_render = self._render(viewer, next_render)
                    pacer.wait(self.data.time)
        finally:
            cmd_sub.close()
            state_pub.close()

    # HELPERS

    @contextlib.contextmanager
    def _viewer(self):
        """Passive viewer with a fixed camera on the robot, or None when headless.

        On exit it waits for the viewer's render thread to finish; Python exiting while that
        thread is still tearing down its window segfaults.
        """
        if self.headless:
            yield None
            return
        # Passive sync applies GUI actions and clears applied forces, even with
        # state_only=True. A watch-only viewer therefore needs separate model/data.
        initial_state = None
        if self.readonly_viewer:
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            initial_state = np.empty(mujoco.mj_stateSize(self.model, state_spec))
            mujoco.mj_getState(self.model, self.data, initial_state, state_spec)
            self._display_model = copy.copy(self.model)
            self._display_data = mujoco.MjData(self._display_model)
            mujoco.mj_copyData(self._display_data, self._display_model, self.data)
        display_model = self._display_model if self.readonly_viewer else self.model
        display_data = self._display_data if self.readonly_viewer else self.data
        # launch, noting the render thread the viewer starts
        before = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(display_model, display_data,
                                              show_left_ui=False, show_right_ui=False)
        render_threads = [t for t in threading.enumerate() if t not in before]
        try:
            # camera 3 m out, looking at pelvis height
            with viewer.lock():
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -20
                viewer.cam.distance = 3.0
                viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
            if self.readonly_viewer:
                # Drain startup GUI actions while physics is still frozen, then
                # display the intended initial state before the trial handshake.
                viewer.sync(state_only=True)
                with viewer.lock():
                    mujoco.mj_copyData(self._display_data, self._display_model, self.data)
                viewer.sync(state_only=True)
                current_state = np.empty_like(initial_state)
                mujoco.mj_getState(self.model, self.data, current_state, state_spec)
                if not np.array_equal(current_state, initial_state):
                    raise RuntimeError("Watch-only viewer mutated the initial physics state")
            yield viewer
        finally:
            # close, then wait for the render thread to exit
            viewer.close()
            for t in render_threads:
                t.join(timeout=2.0)
            self._display_model = self._display_data = None

    def draw_overlays(self, scn: mujoco.MjvScene):
        """Append the reference ball and the commanded and actual twist arrows to `scn`, as in mjlab."""
        # reference ball resting on the floor, if the controller sent one
        if self.ref is not None:
            add_sphere(scn, [self.ref[0], self.ref[1], REF_RADIUS], REF_RADIUS, REF_RGBA)

        # pelvis frame; arrows are body-frame vectors drawn from above the head
        pos = self.data.xpos[self.pelvis_id]
        mat = self.data.xmat[self.pelvis_id].reshape(3, 3)
        start = pos + mat @ np.array([0.0, 0.0, ARROW_HEIGHT])
        lin_b = mat.T @ self.data.qvel[0:3]   # free-joint linear velocity is world frame
        wz_b = self.data.qvel[5]              # free-joint angular velocity is already body frame

        # commanded twist, then actual twist
        for vec_b, rgba in (([self.cmd[0], self.cmd[1], 0.0], CMD_RGBA),
                            ([0.0, 0.0, self.cmd[2]], CMD_RGBA),
                            ([lin_b[0], lin_b[1], 0.0], ACT_RGBA),
                            ([0.0, 0.0, wz_b], ACT_RGBA)):
            add_arrow(scn, start, start + mat @ (np.asarray(vec_b) * ARROW_SCALE), rgba)

    def _render(self, viewer, next_render: float) -> float:
        """Sync the viewer if a frame is due; returns when the next one is due."""
        
        # at most RENDER_HZ frames per second
        now = time.perf_counter()
        if viewer is None or now < next_render:
            return next_render
        with viewer.lock():
            if self.readonly_viewer:
                mujoco.mj_copyData(self._display_data, self._display_model, self.data)
            viewer.user_scn.ngeom = 0
            self.draw_overlays(viewer.user_scn)
        # set_texts can wait on the render thread, so keep it outside viewer.lock.
        self.draw_legend(viewer, now)
        if self.readonly_viewer:
            viewer.sync(state_only=True)
        else:
            viewer.sync()
        return max(next_render + 1.0 / RENDER_HZ, now)

    def draw_legend(self, viewer, now: float):
        """Timing text top-left, as in deploy_robot's simulation, and a key to the overlays top-right."""
        # real-time rate (RTR): sim seconds advanced per wall second since the last frame (1.0 = real time)
        last_sim, last_wall = self.rtr_last
        rtr = (self.data.time - last_sim) / (now - last_wall) if now > last_wall else 0.0
        self.rtr_last = (self.data.time, now)

        timing = (
            LEGEND_FONT,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"Sim time:    {self.data.time:.2f}s\n"
            f"Real time:   {now - self.wall_start:.2f}s\n"
            f"RTR:         {rtr:.2f}\n"
            f"Sim rate:    {1.0 / self.model.opt.timestep:.0f} Hz\n"
            f"Render rate: {RENDER_HZ:.0f} Hz",
            "",
        )

        # what the overlay colors and directions mean
        key = (
            LEGEND_FONT,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "Blue arrows:\nRed arrows:\nFlat / up arrow:\nOrange ball:",
            "commanded twist\nactual twist\nvx, vy / wz\nreference",
        )
        viewer.set_texts([timing, key])

    def save_log(self):
        """Write full-order and ROM logs under data/data/<log_name>/."""
        
        if not self.log_q:
            print("Nothing logged; no file written.")
            return
        
        # one folder per run under data/data/
        out_dir = os.path.join(ROOT_DIR, "data", "data", self.log_name)
        os.makedirs(out_dir, exist_ok=True)
        full_path = os.path.join(out_dir, "g1_29dof_data.npz")
        rom_path = os.path.join(out_dir, "g1_rom_data.npz")

        # preserve the recording schema's (B, T, dim) batch dimension
        q = np.stack(self.log_q)[None].astype(np.float32)
        v = np.stack(self.log_v)[None].astype(np.float32)
        cmd = np.stack(self.log_cmd)[None].astype(np.float32)
        meta = dict(sim_dt=float(self.model.opt.timestep), control_dt=float(self.cfg.control_dt),
                    log_dt=float(self.log_dt))

        # full-order file, then reduced-order file
        np.savez(full_path, q_log=q, v_log=v, cmd_log=cmd, **meta)
        np.savez(rom_path, q_rom_log=q_to_rom(q), v_rom_log=v_to_rom(q, v), cmd_log=cmd, **meta)
        print(f"Saved {q.shape[1]} samples at {1.0 / self.log_dt:g} Hz to {out_dir}/ "
              "(g1_29dof_data.npz, g1_rom_data.npz).")


############################################################################
# MAIN
############################################################################

def main():

    # parse command-line args
    parser = argparse.ArgumentParser(description="Single-robot G1 simulation, driven by a control_*.py.")
    parser.add_argument("--log", default=None, metavar="NAME",
                        help="save the run to data/data/NAME/ when the sim exits")
    parser.add_argument("--log_hz", type=float, default=LOG_HZ, metavar="HZ",
                        help=f"logging rate; must divide the {1.0 / SIM_DT:g} Hz sim rate (default {LOG_HZ:g})")
    args = parser.parse_args()

    # build the sim and run until the viewer closes or Ctrl+C
    robot_cfg = G1Config()
    sim = Simulation(XML_PATH, POLICY_PATH, SIM_DT, robot_cfg, log_name=args.log, log_hz=args.log_hz)
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # save even after Ctrl+C
        if args.log is not None:
            sim.save_log()


if __name__ == "__main__":
    main()
