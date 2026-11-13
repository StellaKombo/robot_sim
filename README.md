# robot_sim
Forward simulation of humanoid robot under a trained RL policies.

## Overview
- `models/` — MuJoCo XML and meshes, one folder per robot: `g1_23dof/` (walk policy) and `g1_29dof/` (jog policy).
- `policy/` — one folder per policy: `walk/` (23-DoF walk checkpoint `g1_23dof_vel.pt`, robot config `G1Config`, and `walk_actor.py` used by both sims) and `jog/` (29-DoF jog ONNX policy and its onnxruntime loader `jog_actor.py`).
- `sim/single/` — single-robot CPU simulation:
  - `simulation_base.py` — shared sim: physics, PD loop, viewer, ZeroMQ command loop and logging.
  - `simulation_walk.py` / `simulation_jog.py` — the 23-DoF walk and 29-DoF jog sims built on it; each runs its
    policy and takes twist commands over ZeroMQ in real time.
  - `control_*.py` — twist-command controllers for either sim: `control_kb.py` (keyboard), `control_joy.py`
    (joystick), `control_custom_walk.py` / `control_custom_jog.py` (your own logic, one per sim), all built on
    `control_base.py`
    (see [Asynchronous simulation and controller](#asynchronous-simulation-and-controller)).
- `sim/parallel/` — GPU-batched MuJoCo Warp rollouts that generate full-order and reduced-order state datasets: `parallel_sim_walk.py` (23-DoF walk: `g1_23dof_data_XX.npz`, `g1_rom_data_XX.npz`) and `parallel_sim_jog.py` (29-DoF jog: `g1_29dof_jog_data_XX.npz`, `g1_rom_jog_data_XX.npz`).
- `utils/` — helpers shared across the code base.
- `KCF_MPC/` — git submodule ([StellaKombo/KCF_MPC](https://github.com/StellaKombo/KCF_MPC)): the trained KCF model and
  the Koopman MPC that `control_custom_walk.py` runs.
- `data/` — generated rollout datasets and the plotting scripts:
  - `plot_23dof_data.py` — plots full-order trajectories and replays selected trajectories in a looping MuJoCo viewer.
  - `plot_rom_data.py` — plots reduced-order planar state and the body-frame cmd overlay, plus a looping 2D top-down animation.

---

## Install
The prerequisites are [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), `make`, and
[git-lfs](https://git-lfs.com) (the KCF model weights in the `KCF_MPC` submodule are stored with it). Clone with the submodule:
```bash
git clone --recurse-submodules <this repo's URL>
```
The environment is defined in [`environment.yml`](environment.yml). From the repo root, run:
```bash
make install
conda activate robot_sim
```
This fetches the `KCF_MPC` submodule if it's missing, creates the `robot_sim` env, and sets `ROBOT_SIM_ROOT_DIR` to this
repo, scoped to the env. To move `KCF_MPC` to the latest commit on its `main`, run
`git submodule update --remote KCF_MPC`, then commit the new pointer.
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
A sim (`simulation_walk.py` or `simulation_jog.py`) and a controller run as two separate processes that talk
over ZeroMQ in real time. The sim owns everything about the robot: physics, the policy, the PD loop, the viewer and logging.
A controller only sends the body-frame twist command `[vx, vy, wz]`. With no controller running,
the sim uses a zero command, so the robot steps in place.

You will open two terminals, each in the `robot_sim` conda environment (`conda activate robot_sim`).
In the first terminal, you will launch the simulation. In the second terminal, you will launch the
controller of your choice. The controller can be stopped and restarted while the sim keeps running.

Terminal 1 (pick one):
```bash
python sim/single/simulation_walk.py   # 23-DoF walk
python sim/single/simulation_jog.py    # 29-DoF jog
```
Terminal 2 (pick one):
```bash
python sim/single/control_kb.py             # keyboard (walk or jog sim)
python sim/single/control_joy.py            # joystick (walk or jog sim)
python sim/single/control_custom_walk.py    # your own logic, walk sim (Koopman MPC)
python sim/single/control_custom_jog.py     # your own logic, jog sim started with --raw_cmd
```

#### Choosing the command source
- `control_kb.py`: arrow keys in a small window. Up/Down is `vx`, Left/Right is `wz`, Shift + Left/Right
  is `vy`, and `+`/`-` set the speed from 0.1x to 1.0x. Keys only register while the window has focus.
  It turns red when it doesn't, so click it.
- `control_joy.py`: Xbox controller. Left stick is `vx`, `vy`, and the right stick is `wz`.
- `control_custom_walk.py`: your own logic in `CustomCommand.get(state)`, for `simulation_walk.py`. Copy it to
  add another controller.
- `control_custom_jog.py`: your own logic in `CustomJogCommand.get(state)`, for `simulation_jog.py --raw_cmd`.
  It sends jog twists directly, so the sim must not refit them as stick positions. It is the same PD on the
  single-integrator planar model as in the walk controller, tracking a Lissajous curve scaled for a unicycle
  jog (a = 7.5 m, b = 3.75 m, w = 0.12 rad/s: speed 0.60–1.27 m/s, `|wz|` ≤ 0.38 rad/s, one 52 s loop).
  The jog sim drops `vy`.
- Each custom controller stops (walk) or sends zero with a warning (jog) when paired with the other sim.

`state` holds sim time `t`, the robot's full `q` and `v`, and `prev_action`, the policy action (23-D walk, 29-D jog) from
the previous 50 Hz policy tick (the one in the policy's next observation). In `v`, the base's linear
velocity is in the world frame and its angular velocity is in the body frame. `control_custom_walk.py` also
builds the 84-D closed-loop state of the full-order KCF model each tick (`x_kcf` in `get`); that KCF model
is trained on walk data.

#### Commands in the jog sim
The jog policy only knows its trained twists: forward `vx` 0.5–1.5 or backward −1.0 to −0.5 (each with `wz`
±0.5), turning in place with `|wz|` 1–2, or standing; never `vy`. `simulation_jog.py` therefore reads a
controller's twist as stick positions (full scale = the walk `cmd_scale`, 0.75) and fits them into those
bands, as deploy_robot's jog controller does: a small `vx` and `wz` stand, `vx` picks forward/backward and
its speed, `wz` steers on top, `wz` alone turns in place, and `vy` is ignored. So the keyboard and joystick
controllers work unchanged. For a controller that already sends jog twists, pass `--raw_cmd`; the twist is
then only clipped to the policy's range.

> [!IMPORTANT]
> `control_custom_jog.py` (and any controller that sends real velocities in m/s and rad/s) needs the jog sim
> started with `--raw_cmd`:
> ```bash
> python sim/single/simulation_jog.py --raw_cmd
> python sim/single/control_custom_jog.py
> ```
> Without it the sim treats those velocities as stick positions and distorts them (e.g. 0.9 m/s becomes
> 1.5 m/s), and tracking gets several times worse. The keyboard and joystick need the default, *without*
> `--raw_cmd`.

#### Saving a run
Pass `--log <name>` to the sim. When it exits, it writes to `data/data/<name>/`, in the same format and with
the same names as the GPU datasets: `g1_23dof_data.npz` and `g1_rom_data.npz` from the walk sim,
`g1_29dof_jog_data.npz` and `g1_rom_jog_data.npz` from the jog sim. Reusing a name overwrites that run.
Plot it with `python data/plot_rom_data.py <name> --walk` (or `--jog` for a jog run; see below).
The state is logged at 100 Hz by default. Set another rate with `--log_hz`, which must divide the
500 Hz sim rate (e.g. 500, 250, 100, 50).
```bash
python sim/single/simulation_walk.py --log lissajous --log_hz 200   # logged at 200 Hz
```

#### Behavior to know
- Both scripts must run on the same PC. The sim takes commands on port 5555 and sends its state on
  port 5556, both on localhost only, so other machines can't connect.
- Commands are sent every 20 ms. If none arrives for half a second (`CMD_TIMEOUT` in `simulation_base.py`), the sim
  switches to a zero command until the controller is back.
- If the sim falls behind real time, it runs without sleeping until it catches up.
- Press `C` in the viewer to lock the camera onto the pelvis; press it again to free the camera. The camera
  follows the pelvis's x and y at a fixed height (the pelvis height at the start), so the view doesn't bob with
  each stride, and it keeps your orbit and zoom.
- Timing affects the result, so two runs never match exactly.

### Generate datasets (GPU)
Runs batched MuJoCo Warp rollouts and writes both the full-order state and the
reduced-order state for each rollout into `./data/data/`. Requires a CUDA GPU.
```bash
python sim/parallel/parallel_sim_walk.py               # logged every physics step (500 Hz)
python sim/parallel/parallel_sim_walk.py --log_hz 100  # logged at 100 Hz
```
Edit the constants in the `__main__` block of [`sim/parallel/parallel_sim_walk.py`](sim/parallel/parallel_sim_walk.py)
(`batch_size`, `sim_dt_des`, `duration`, `N_datasets`, `cmd_zoh_steps`) to control the
rollout size and command randomization. Each iteration produces two files:
- `g1_23dof_data_XX.npz` — full order: `q_log`, `v_log`, `cmd_log`.
- `g1_rom_data_XX.npz`   — reduced order: `q_rom_log` `[px, py, theta]`,
  `v_rom_log` `[vx, vy, wz]` (body frame), `cmd_log`.

### Visualize datasets
Each script takes an optional run name. With one, it plots the run that a sim's `--log <name>`
saved in `data/data/<name>/`. Without one, they plot GPU dataset `_01`. Edit the `num_trajectories` or
`seed` constants at the top of each `main()` to change which trajectories are shown.

Full-order — Matplotlib plots of `q` and `v` plus a looping MuJoCo replay of the
selected trajectories:
```bash
python data/plot_23dof_data.py            # GPU dataset _01
python data/plot_23dof_data.py lissajous  # the run saved with --log lissajous
python data/plot_29dof_data.py            # jog GPU dataset _01 (--dataset N for another)
python data/plot_29dof_data.py myjog      # a simulation_jog.py run saved with --log myjog
```

Reduced-order — Matplotlib plots of `q_rom` / `v_rom` (with body-frame cmd
overlay) plus a looping 2D top-down animation with heading triangles:
```bash
python data/plot_rom_data.py --walk              # walk GPU dataset _01 (--dataset N for another)
python data/plot_rom_data.py lissajous --walk    # the walk run saved with --log lissajous
python data/plot_rom_data.py --jog               # jog GPU dataset _01 (--dataset N for another)
python data/plot_rom_data.py myjog --jog         # a simulation_jog.py run
```
