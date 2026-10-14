# robot_sim
Forward simulation of humanoid robot under a trained RL policies.

## Overview
- `models/` — MuJoCo XML and meshes for the G1 23-DoF humanoid.
- `policy/` — trained policy checkpoint (`g1_23dof_vel.pt`), robot config (`G1Config`), and the `actor` used by both sims.
- `sim/single/` — `single_sim.py`, a real-time, single-environment closed-loop MuJoCo viewer with optional joystick command input.
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

### Real-time simulation
Drives the trained policy in a single MuJoCo viewer on CPU. If a joystick is
plugged in at startup it will drive the velocity command; otherwise the command
defaults to zero (or to `--cmd` if provided).
```bash
python sim/single/single_sim.py                          # cmd = [0, 0, 0]
python sim/single/single_sim.py --cmd 0.5 0.0 0.0        # walk forward at 0.5 m/s
python sim/single/single_sim.py --cmd 0.0 0.0 0.5        # spin in place at 0.5 rad/s
```
`--cmd` takes three floats `vx vy omega_z` in the **body frame**; with a
joystick connected it acts as the initial value only.

Other overrides:
```bash
python sim/single/single_sim.py --xml models/g1_23dof.xml \
                         --policy policy/g1_23dof_vel.pt \
                         --sim_dt 0.002
```

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
