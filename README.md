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

---

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

---

## Usage
Scripts find the repo through the `ROBOT_SIM_ROOT_DIR` variable, which `make install` sets in the
conda env, so they can be run from any directory. Outside the env they fall back to their own location.

### Asynchronous simulation and controller
`simulation.py` and a controller run as two separate processes that talk over ZeroMQ in real time.
`simulation.py` owns everything about the robot: physics, the policy, the PD loop, the viewer and logging.
A controller only sends the body-frame twist command `[vx, vy, wz]`. With no controller running,
the sim uses a zero command, so the robot steps in place.

You will open two terminals, each in the `robot_sim` conda environment (`conda activate robot_sim`).
In the first terminal, you will launch the simulation. In the second terminal, you will launch the
controller of your choice. The controller can be stopped and restarted while the sim keeps running.

Terminal 1:
```bash
python sim/single/simulation.py
```
Terminal 2 (pick one):
```bash
python sim/single/control_kb.py        # keyboard
python sim/single/control_joy.py       # joystick
python sim/single/control_custom.py    # your own logic
```

#### Choosing the command source
- `control_kb.py`: arrow keys in a small window. Up/Down is `vx`, Left/Right is `wz`, Shift + Left/Right
  is `vy`, and `+`/`-` set the speed from 0.1x to 1.0x. Keys only register while the window has focus.
  It turns red when it doesn't, so click it.
- `control_joy.py`: Xbox controller. Left stick is `vx`, `vy`, and the right stick is `wz`.
- `control_custom.py`: your own logic in `CustomCommand.get(state)`. Copy it to add another controller.

`state` holds sim time `t`, the robot's full `q` and `v`, and `prev_action`, the 23-D policy action from
the previous 50 Hz policy tick (the one in the policy's next observation). In `v`, the base's linear
velocity is in the world frame and its angular velocity is in the body frame. `control_custom.py` also
builds the 84-D closed-loop state of the full-order KCF model each tick (`x_kcf` in `get`).

#### Saving a run
Pass `--log <name>` to `simulation.py`. When the sim exits, it writes `data/data/<name>/g1_23dof_data.npz`
and `data/data/<name>/g1_rom_data.npz`, in the same format and with the same names as the GPU datasets.
Reusing a name overwrites that run. Plot it with `python data/plot_rom_data.py exp` (see below).
The state is logged at 100 Hz by default. Set another rate with `--log_hz`, which must divide the
500 Hz sim rate (e.g. 500, 250, 100, 50).
```bash
python sim/single/simulation.py --log lissajous --log_hz 200   # loagged at 200 Hz
```

#### Behavior to know
- Both scripts must run on the same PC. The sim takes commands on port 5555 and sends its state on
  port 5556, both on localhost only, so other machines can't connect.
- Commands are sent every 20 ms. If none arrives for half a second (`CMD_TIMEOUT` in `simulation.py`), the sim
  switches to a zero command until the controller is back.
- If the sim falls behind real time, it runs without sleeping until it catches up.
- Timing affects the result, so two runs never match exactly.

### Generate datasets (GPU)
Runs batched MuJoCo Warp rollouts and writes both the full-order state and the
reduced-order state for each rollout into `./data/data/`. Requires a CUDA GPU.
```bash
python sim/parallel/parallel_sim.py               # logged every physics step (500 Hz)
python sim/parallel/parallel_sim.py --log_hz 100  # logged at 100 Hz
```
Edit the constants in the `__main__` block of [`sim/parallel/parallel_sim.py`](sim/parallel/parallel_sim.py)
(`batch_size`, `sim_dt_des`, `duration`, `N_datasets`, `cmd_zoh_steps`) to control the
rollout size and command randomization. Each iteration produces two files:
- `g1_23dof_data_XX.npz` — full order: `q_log`, `v_log`, `cmd_log`.
- `g1_rom_data_XX.npz`   — reduced order: `q_rom_log` `[px, py, theta]`,
  `v_rom_log` `[vx, vy, wz]` (body frame), `cmd_log`.

### Visualize datasets
Both scripts take an optional run name. With one, they plot the run that `simulation.py --log <name>`
saved in `data/data/<name>/`. Without one, they plot GPU dataset `_01`. Edit the `num_trajectories` or
`seed` constants at the top of each `main()` to change which trajectories are shown.

Full-order — Matplotlib plots of `q` and `v` plus a looping MuJoCo replay of the
selected trajectories:
```bash
python data/plot_23dof_data.py            # GPU dataset _01
python data/plot_23dof_data.py lissajous  # the run saved with --log lissajous
```

Reduced-order — Matplotlib plots of `q_rom` / `v_rom` (with body-frame cmd
overlay) plus a looping 2D top-down animation with heading triangles:
```bash
python data/plot_rom_data.py            # GPU dataset _01
python data/plot_rom_data.py lissajous  # the run saved with --log lissajous
```
