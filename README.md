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
- `KCF_MPC/` — git submodule ([StellaKombo/KCF_MPC](https://github.com/StellaKombo/KCF_MPC)): the trained KCF model and
  the Koopman MPC that `control_custom.py` runs.
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
python sim/single/control_custom.py --lissajous  # projected 85-state KCF prediction
python sim/single/control_custom.py --single-integrator --lissajous  # ideal-model comparison
```

#### Choosing the command source
- `control_kb.py`: arrow keys in a small window. Up/Down is `vx`, Left/Right is `wz`, Shift + Left/Right
  is `vy`, and `+`/`-` set the speed from 0.1x to 1.0x. Keys only register while the window has focus.
  It turns red when it doesn't, so click it.
- `control_joy.py`: Xbox controller. Left stick is `vx`, `vy`, and the right stick is `wz`.
- `control_custom.py`: KCF-MPC control. It holds the fixed target by default; add `--lissajous` to track
  the configured Lissajous pose-and-velocity reference. The default checkpoint is
  `KCF_MPC/results/g1_kcf_handoff_epoch01540/model_final.pt`. It predicts all 85 native channels,
  applies its trained kinematic projection, and re-lifts the corrected state at every step.
  Its fixed local frame is mandatory and applied automatically. Use
  `--single-integrator` for the separate ideal body-twist kinematic baseline.

`state` holds sim time `t`, the robot's full `q` and `v`, and `prev_action`, the 23-D policy action from
the previous 50 Hz policy tick (the one in the policy's next observation). In `v`, the base's linear
velocity is in the world frame and its angular velocity is in the body frame.

The KCF controller is separated into these modules:

- `sim/single/control_custom.py`: command-source entry point and existing CLI flags.
- `utils/kcf_controller.py`: MPC startup, feedback, execution, buffering, and trace lifecycle.
- `utils/kcf_reference.py`: fixed/Lissajous paths and tracked reference construction.
- `utils/g1_kcf_state.py`: NumPy-only encoder for the native 85-state feedback, reusable without loading a controller.
- `utils/g1_kcf_state_frame.py`: KCF-schema adapter using the shared geometry in `utils/relative_frame.py`.

To run the existing synchronous path, which pauses physics during each solve:

```bash
CUDA_VISIBLE_DEVICES=0 python KCF_MPC/mpc/visual_recovery_trial.py new_kcf_trial \
  --synchronous --headless
```

Use a fresh run name. Omitting `--headless` opens the viewer. Trial profiles retain their
existing settings; the default trial08 profile has gtol=0.003, while direct
`control_custom.py` uses gtol=0.0015.

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
The default remains the original uniform ZOH sampler, including its seed and command
hold behavior. Use `--batch-size`, `--duration`, `--datasets`, `--seed`, and
`--cmd-zoh-steps` to change collection size without editing the script. Physics stays
at 0.002 s and the existing policy/PD controller are unchanged.

Expanded collection is opt-in and also saves under `data/data/` by default:
```bash
python -u sim/parallel/parallel_sim.py --command-profile mixed \
  --batch-size 100 --duration 6 --datasets 1 --cmd-zoh-steps 50 \
  --command-scale 0.5 --seed 2
```
The default mix is 30% `constant_zoh` (one command for the entire trajectory),
25% `multi_step_zoh`, 20% `smooth`, 15% `multisine`, and 10% `reversals`.
`--profile-weights` accepts five proportions in that order; each profile can also
be selected individually. Forward/backward, lateral, turning, combined, and
zero-command families are distributed within each profile. Labels describe the
intended excitation, not guaranteed achieved motion. `--command-scale` narrows
the existing policy command box; it cannot expand it.

Each iteration produces full-order and ROM NPZ files with the original keys.
Profile and collection settings distinguish new files without modifying or
overwriting existing ZOH recordings:

- `g1_23dof_data_mixed_B100_T3001_zoh50_seed2_01.npz` — full order: `q_log`, `v_log`, `cmd_log`.
- `g1_rom_data_mixed_B100_T3001_zoh50_seed2_01.npz` — reduced order: `q_rom_log` `[px, py, theta]`,
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

Reduced-order — Matplotlib plots of `[px, py, sin(theta), cos(theta)]` and `v_rom`
(with body-frame cmd overlay), plus a looping 2D top-down animation with heading triangles:
```bash
python data/plot_rom_data.py            # GPU dataset _01
python data/plot_rom_data.py lissajous  # the run saved with --log lissajous
```

The saved ROM logger still uses `[px, py, theta]`; plotting derives the sine/cosine
pair without changing old files. This matches the MPC heading features. A wrapped
heading-error angle in degrees is still a valid diagnostic, not a legacy yaw-state cost.

To replay two controller runs together at real-time speed, independently of their
original solver speeds (from `robot_sim`):

```bash
python data/compare_rom_data.py kcf_01 si_01
```

This opens overlaid ROM playback on one shared axis, with measured heading triangles,
trails, and hollow reference triangles. Both recordings advance on one monotonic
clock: one simulated second per wall-clock second. Close the playback window to stop.
The static comparison shows sine/cosine heading, world position, body velocity,
applied commands and the reference reconstructed from each saved `trial_plan.json`.
It does not rerun physics or the controllers and never substitutes today's default
path for missing saved metadata. Changed-source/incomplete runs and mismatched
references/initial poses are flagged as warnings, not silently treated as a fair benchmark.

The two positional arguments can also be run directories or ROM NPZ files.
Overlay is the default (`--overlay` remains accepted). Use `--side-by-side` for
separate axes, `--once` for one pass,
`--speed 0.5` for half speed, and `--start 10 --end 35` for an interval. Default
speed is `1.0`, render rate is 50 Hz, and environment index is zero (`--env-index`).
Different logging periods are synchronized by simulation time; playback stops at
the shorter recording's end and skips obsolete frames if rendering falls behind.

Save the new plots alongside the tuning comparison, without a GUI:

```bash
MPLBACKEND=Agg python data/compare_rom_data.py kcf_01 si_01 \
  --no-playback --output-dir KCF_MPC/reports/mpc_tuning_sincos_v1
```

This writes `rom_states.png` and `rom_playback_preview.png`. For reuse in another
plotting script, `compare_rom_data.py` exposes `plot_rom_comparison(...)`,
`playback_rom_comparison(...)`, and `ROMComparisonPlayback.update(sim_time)`.

### Inspect RL policy outputs versus MPC commands

The ROM velocity plots show achieved base motion, not the RL network's outputs.
The trial records already contain those outputs in `applied_policy_commands.npz`.
To compare both controllers without rerunning simulation:

```bash
python data/plot_policy_actions.py kcf_01 si_01 --start 10 --zoom 6
```

This opens three diagnostic figures: MPC commands / raw RL actions / desired and
measured joint angles / per-tick target changes; frequency spectra; and a summary
of variation across all 23 joints. Blue is the first run, red the second. The
default time zoom shows 10–16 s; spectra and summary use 10 s through the last
common recorded RL output. `--end` restricts that analysis interval. Choose other
joints with `--joints right_hip_pitch right_knee right_ankle_pitch` or numeric
zero-based indices (one to six joints). No smoothing is applied to the time traces.

The loader shifts `previous_policy_action[t]` to its actual output time
`t-control_dt`, excludes the artificial startup history, and aligns the commands
and measured joint angles accordingly. It reconstructs desired angles from the
policy action scale plus nominal joint pose, verifying saved configuration hashes
when available. A changed historical policy config is rejected rather than silently
using wrong scaling. The final policy output is not present in the previous-action
log, so the analysis ends slightly before the full simulation duration.

Look for irregular tick-to-tick jitter, excessive target changes relative to SI,
and extra spectral power between gait-related peaks. The known 0.6 s gait rhythm
(about 1.67 Hz) and harmonics are expected motion, not automatically noise. Likewise,
the shaded frequency band (`--cutoff-hz 5`) is an illustrative diagnostic, not a
noise threshold. Contacts and nonlinear feedback can also produce higher frequencies;
these plots alone do not prove that MPC is the cause of a fluctuation.

Save headlessly alongside the MPC comparison:

```bash
MPLBACKEND=Agg python data/plot_policy_actions.py kcf_01 si_01 \
  --output-dir KCF_MPC/reports/mpc_tuning_sincos_v1/rl_actions --no-show
```

This writes `policy_time_traces.png`, `policy_spectra.png`,
`policy_joint_summary.png`, and `policy_metrics.json`. It does not change any
controller weights, policy outputs, filters or simulation logging.

### Offline MPC-data fine-tuning

For the optional Muon/AdamW workflow with original-data replay, four MPC reference
families, held-out prediction checks and explicit control validation, see
[`KCF_MPC/README.md`](KCF_MPC/README.md#conservative-mpc-data-fine-tuning-offline-opt-in).
The original prepared dataset can remain on its training server, or be regenerated
from the published epoch-1540 Git LFS sources. Collection and adaptation write
fresh directories and never automatically replace the configured checkpoint.
