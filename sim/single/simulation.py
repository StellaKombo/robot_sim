##
#
#  Simulation
#
##

# standard imports
import argparse
import contextlib
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
from utils.zmq_utils import CMD_PORT, bind_sub, unpack_twist

# GLOBAL VARIABLES
XML_PATH = os.path.join(ROOT_DIR, "models", "g1_23dof.xml")
POLICY_PATH = os.path.join(ROOT_DIR, "policy", "g1_23dof_vel.pt")
SIM_DT = 0.002      # [s] physics timestep
RENDER_HZ = 50.0    # [Hz] visual rendering frequency
CMD_TIMEOUT = 0.5   # [s] without a command from a controller before falling back to zero


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
                 headless: bool = False, log_path: str = None):
        # config and run options
        self.cfg = robot_cfg or G1Config()
        self.nu = len(self.cfg.Kp)
        self.headless = headless
        self.log_path = log_path

        # model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        
        # pelvis gyro's slot in sensordata, looked up by name
        gyro_adr = self.model.sensor("pelvis_imu_gyro_sensor").adr[0]
        self.gyro_slice = slice(gyro_adr, gyro_adr + 3)

        # physics steps per policy query
        ratio = self.cfg.control_dt / sim_dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError(f"sim_dt ({sim_dt}) must divide control_dt ({self.cfg.control_dt}) exactly.")
        self.decimation = int(round(ratio))

        # policy
        self.device = torch.device("cpu")
        self.policy = load_policy(policy_path, self.device)
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
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.action = self.prev_action.copy()
        self.log_q, self.log_v, self.log_cmd = [], [], []
        
        # recompute kinematics and sensors for the new state
        mujoco.mj_forward(self.model, self.data)

    # COMMAND

    def set_cmd(self, cmd: np.ndarray):
        """Body-frame [vx, vy, wz], clipped to the policy's training range."""
        self.cmd = np.clip(np.asarray(cmd, dtype=np.float32), -self.cmd_scale, self.cmd_scale)

    # POLICY

    def build_observation(self) -> torch.Tensor:
        """80-dim policy observation, in the exact layout the policy was trained on."""
        
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
            qpos_j - self.default_j,    # 11:34
            qvel_j,                     # 34:57
            self.prev_action,           # 57:80
        ]).astype(np.float32)
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    def policy_step(self):
        """One policy query; the action is held until the next one."""
        # run the policy; the action also feeds the next observation
        self.action = self.policy(self.build_observation()).squeeze(0).cpu().numpy().astype(np.float32)
        self.prev_action = self.action

    # PHYSICS

    def step(self):
        """One physics step: PD torque toward the held action's joint targets, mj_step, log."""
        
        if self.log_path is not None and not self.log_q:
            self._record()   # initial state, logged with the first command like parallel_sim
        
        # PD torque toward the joint targets encoded by the held action
        qpos_des = self.action * self.cfg.action_scale + self.cfg.default_joint_pos
        qpos_j = self.data.qpos[7:7 + self.nu]
        qvel_j = self.data.qvel[6:6 + self.nu]
        self.data.ctrl[:] = self.cfg.Kp * (qpos_des - qpos_j) + self.cfg.Kd * (-qvel_j)
        
        # advance physics, then log the new state
        mujoco.mj_step(self.model, self.data)
        if self.log_path is not None:
            self._record()

    def _record(self):
        # full state plus the command in effect
        self.log_q.append(self.data.qpos.copy())
        self.log_v.append(self.data.qvel.copy())
        self.log_cmd.append(self.cmd.copy())

    # RUN LOOP

    def run(self, duration: float = None):
        """Real time until the viewer closes; reads the newest command before each policy query."""
        
        # command socket, optional step budget, and real-time pacer
        ctx = zmq.Context.instance()
        cmd_sub = bind_sub(ctx, CMD_PORT)
        n_steps = None if duration is None else int(round(duration / self.model.opt.timestep))
        pacer = Pacer()
        print(f"Listening for commands on port {CMD_PORT}; using zero command until a controller connects.")

        # whether commands are arriving, and when the last one came
        connected = False
        last_rx = None
        try:
            # loop until the duration ends or the viewer closes
            with self._viewer() as viewer:
                pacer.start(self.data.time)
                next_render = time.perf_counter()
                k = 0
                while (n_steps is None or k < n_steps) and (viewer is None or viewer.is_running()):
        
                    # policy tick: read the newest command, check the timeout, query the policy
                    if k % self.decimation == 0:
                        try:
                            self.set_cmd(unpack_twist(cmd_sub.recv(zmq.NOBLOCK)))
                            last_rx = time.perf_counter()
                            if not connected:
                                print("Receiving commands from the controller.")
                                connected = True
                        except zmq.Again:
                            pass
        
                        # controller went quiet: fall back to zero
                        if connected and time.perf_counter() - last_rx > CMD_TIMEOUT:
                            print(f"No command for {CMD_TIMEOUT} s; using zero command.")
                            connected = False
                            self.set_cmd(np.zeros(3))
                        self.policy_step()

                    # every physics step: PD and physics, render if due, keep real time
                    self.step()
                    k += 1
                    next_render = self._render(viewer, next_render)
                    pacer.wait(self.data.time)
        finally:
            cmd_sub.close()

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
        # launch, noting the render thread the viewer starts
        before = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                              show_left_ui=False, show_right_ui=False)
        render_threads = [t for t in threading.enumerate() if t not in before]
        
        # camera 3 m out, looking at pelvis height
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
        try:
            yield viewer
        finally:
            # close, then wait for the render thread to exit
            viewer.close()
            for t in render_threads:
                t.join(timeout=2.0)

    def _render(self, viewer, next_render: float) -> float:
        """Sync the viewer if a frame is due; returns when the next one is due."""
        
        # at most RENDER_HZ frames per second
        now = time.perf_counter()
        if viewer is None or now < next_render:
            return next_render
        viewer.sync()
        return max(next_render + 1.0 / RENDER_HZ, now)

    def save_log(self):
        """Write the run as npz in parallel_sim.py's format, so data/plot_*.py work unchanged."""
        
        if not self.log_q:
            print("Nothing logged; no file written.")
            return
        
        # path stem without .npz, and its folder
        stem = self.log_path[:-4] if self.log_path.endswith(".npz") else self.log_path
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

        # batch dimension of 1, matching parallel_sim's (B, T, dim)
        q = np.stack(self.log_q)[None].astype(np.float32)
        v = np.stack(self.log_v)[None].astype(np.float32)
        cmd = np.stack(self.log_cmd)[None].astype(np.float32)
        meta = dict(sim_dt=float(self.model.opt.timestep), control_dt=float(self.cfg.control_dt))

        # full-order file, then reduced-order file
        np.savez(f"{stem}_23dof.npz", q_log=q, v_log=v, cmd_log=cmd, **meta)
        np.savez(f"{stem}_rom.npz", q_rom_log=q_to_rom(q), v_rom_log=v_to_rom(q, v), cmd_log=cmd, **meta)
        print(f"Saved {q.shape[1]} steps to {stem}_23dof.npz and {stem}_rom.npz.")


############################################################################
# MAIN
############################################################################

def main():

    # parse command-line args
    parser = argparse.ArgumentParser(description="Single-robot G1 simulation, driven by a control_*.py.")
    parser.add_argument("--log", default=None, help="save the run to this npz path stem")
    args = parser.parse_args()

    # build the sim and run until the viewer closes or Ctrl+C
    robot_cfg = G1Config()
    sim = Simulation(XML_PATH, POLICY_PATH, SIM_DT, robot_cfg, log_path=args.log)
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
