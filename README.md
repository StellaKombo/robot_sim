# robot_sim
Forward simulation of humanoid robot under a trained RL policies.

## Overview
- `models/` — MuJoCo XML and meshes for the G1 23-DoF humanoid.
- `policy/` — trained policy checkpoint (`g1_23dof_vel.pt`), robot config (`G1Config`), and the `actor` used by both sims.
- `sim/single/` — single-robot CPU simulation:
  - `simulation.py` — the simulation, which runs the policy and takes twist commands over ZeroMQ in real time.
  - `control_*.py` — twist-command controllers for `simulation.py`: `control_kb.py` (keyboard), `control_joy.py`
    (joystick), `control_custom.py` (your own logic), all built on `control_base.py`
    (see [Asynchronous simulation and controller](#asynchronous-simulation-and-controller)).
- `sim/parallel/` — `parallel_sim.py`, GPU-batched MuJoCo Warp rollouts that generate full-order state (`g1_23dof_data_XX.npz`) and reduced-order state (`g1_rom_data_XX.npz`) datasets.
- `utils/` — helpers shared across the code base.
- `data/` — generated rollout datasets and the plotting scripts:
  - `plot_23dof_data.py` — plots full-order trajectories and replays selected trajectories in a looping MuJoCo viewer.
  - `plot_rom_data.py` — plots reduced-order planar state and the body-frame cmd overlay, plus a looping 2D top-down animation.

## Install
The prerequisites are [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and `make`. The environment is defined in [`environment.yml`](environment.yml). From the repo root, run:
```bash
make install
conda activate robot_sim
```
This creates the `robot_sim` env and sets `ROBOT_SIM_ROOT_DIR` to this repo, scoped to the env.
If `environment.yml` changes, update the existing env in place:
```bash
make update
```
To remove the environment:
```bash
conda deactivate  # if the env is active
make uninstall
```

## Usage
Scripts find the repo through the `ROBOT_SIM_ROOT_DIR` variable, which `make install` sets in the
conda env, so they can be run from any directory. Outside the env they fall back to their own location.

### Asynchronous simulation and controller
`simulation.py` and a controller run as two separate processes that talk over ZeroMQ in real time.
`simulation.py` owns everything about the robot: physics, the policy, the PD loop, the viewer and logging.
A controller only sends the body-frame twist command `[vx, vy, wz]`. With no controller running,
the sim uses a zero command, so the robot steps in place.

You will open two terminals, each in the `robot_sim` conda environment (`conda activate robot_sim`).
In the first terminal, you will launch the controller. In the second terminal, you will launch the
simulation. They can be started in either order, and the controller can be stopped and restarted
while the sim keeps running.

Terminal 1:
```bash
python sim/single/control_kb.py              # or control_joy.py / control_custom.py
```
Terminal 2:
```bash
python sim/single/simulation.py
```
The sim prints a line when commands start arriving and when they stop. To stop, press `Ctrl+C` in
Terminal 1 and close the viewer window in Terminal 2.

#### Choosing the command source
Each controller is its own script:
- `control_joy.py` reads an Xbox controller. Left stick: forward/back is `vx`, left/right is `vy`.
  Right stick: left/right is the yaw rate `wz`. The command is zero until a joystick is plugged in.
- `control_kb.py` opens a small window that reads the arrow keys while it has focus. Up/Down: forward/back
  (`vx`), Left/Right: turn (`wz`), Shift + Left/Right: step sideways (`vy`). A held key commands
  speed x `cmd_scale` on its axis, and releasing it returns that axis to zero. `+` and `-` change the
  speed by 0.1x, between 0.1x and 1.0x, starting at 1.0x. Closing the window stops the controller.
  The window only reads keys while it has focus, and the sim's viewer takes focus when it opens.
  Click the keyboard window after starting the sim. It turns red whenever it doesn't have focus.
- `control_custom.py` calls `CustomCommand.get(state)`, which you fill in with your own logic. It
  returns zero until you do. To add another controller, copy this file and rename it.
```bash
python sim/single/control_kb.py
python sim/single/control_joy.py
python sim/single/control_custom.py
```
`state` holds the time since the controller started and the robot's full position and velocity
vectors. The controller gets no feedback from the sim, so for now the state is always the default
standing pose with zero velocity. The sim clips every command to the policy's training range, `cmd_scale`.

#### Saving a run
Pass `--log <stem>` to `simulation.py`. When the sim exits, it writes `<stem>_23dof.npz` and `<stem>_rom.npz`
in the GPU dataset format, so the plotting scripts below can read them.
```bash
python sim/single/simulation.py --log data/data/run
```

#### Behavior to know
- Both scripts must run on the same PC. The sim listens on port 5555 on localhost only, so other
  machines can't send it commands.
- Commands are sent every 20 ms. If none arrives for half a second (`CMD_TIMEOUT` in `simulation.py`), the sim
  switches to a zero command until the controller is back.
- If the sim falls behind real time, it runs without sleeping until it catches up.
- Timing affects the result, so two runs never match exactly.

### Generate datasets (GPU)
Runs batched MuJoCo Warp rollouts and writes both the full-order state and the
reduced-order state for each rollout into `./data/data/`. Requires a CUDA GPU.
```bash
python sim/parallel/parallel_sim.py
```
Edit the constants at the top of [`sim/parallel/parallel_sim.py`](sim/parallel/parallel_sim.py)
(`batch_size`, `sim_dt_des`, `T`, `N_datasets`, `cmd_zoh_steps`) to control the
rollout size and command randomization. Each iteration produces two files:
- `g1_23dof_data_XX.npz` — full order: `q_log`, `v_log`, `cmd_log`.
- `g1_rom_data_XX.npz`   — reduced order: `q_rom_log` `[px, py, theta]`,
  `v_rom_log` `[vx, vy, wz]` (body frame), `cmd_log`.

### Visualize datasets
Both scripts default to dataset `_01`. Edit the `data_path`, `num_trajectories`,
or `seed` constants at the top of each `main()` to change selection.

Full-order — Matplotlib plots of `q` and `v` plus a looping MuJoCo replay of the
selected trajectories:
```bash
python data/plot_23dof_data.py
```

Reduced-order — Matplotlib plots of `q_rom` / `v_rom` (with body-frame cmd
overlay) plus a looping 2D top-down animation with heading triangles:
```bash
python data/plot_rom_data.py
```
