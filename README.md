# robot_sim
Forward simulation of humanoid robot under a trained RL policies.

The active workflow in this checkout is the **29-DoF jogging simulator**. The
previous walking checkout, including its datasets and KCF checkpoints, is kept
beside this directory at `../robot_sim_walk/`. When running that archived
checkout, set `ROBOT_SIM_ROOT_DIR` to its directory so an existing environment
does not resolve imports or data paths against this jogging checkout.

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
ALternatively run:
```
micromamba activate robot_sim
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

#### Jogging policy logs for KCF training

Both `simulation_jog.py --log <name>` and `parallel_sim_jog.py` save additional
control-rate arrays in the **full-order** `g1_29dof_jog_data*.npz` file. These
records are sampled at every 50 Hz policy query, independently of `--log_hz`.
The existing `q_log`, `v_log`, `cmd_log`, and `mode_log` retain their original
physics/logging-rate meaning. `time_log` timestamps those physics samples.

For `B` robots and `K` policy queries, the new fields are:

| Field | Shape | Meaning |
|---|---|---|
| `policy_time_log` | `(B, K)` | Actual pre-query simulation time |
| `policy_q_log`, `policy_v_log` | `(B, K, 36)`, `(B, K, 35)` | Pre-query robot configuration and velocity |
| `policy_prev_action_log` | `(B, K, 29)` | Action used in this query's observation |
| `policy_cmd_log`, `policy_mode_log` | `(B, K, 3)`, `(B, K)` | Applied twist after clipping/shaping, and its mode |
| `policy_phase_step_log` | `(B, K)` | Actual free-running clock index, including any random initial offset |
| `policy_clock_log` | `(B, K, 2)` | Ungated clock sine/cosine, retained while standing |
| `policy_gait_phase_log` | `(B, K, 2)` | Phase passed to the policy; exactly zero while standing |
| `policy_obs_log` | `(B, K, 98)` | Exact observation passed to ONNX inference |
| `policy_action_log` | `(B, K, 29)` | Output of this query, held during the following control interval |
| `policy_h_in_log`, `policy_c_in_log` | `(B, K, layers, hidden)` | LSTM memory immediately before inference |
| `policy_h_out_log`, `policy_c_out_log` | `(B, K, layers, hidden)` | LSTM memory returned by that inference |

The current LSTM has one layer and 128 hidden and cell channels each. MLP
policies omit recurrent arrays. Query `k` uses `policy_cmd_log[:, k]` to drive
the transition from `policy_q_log[:, k]` / `policy_v_log[:, k]` to the next
query's state. Use these command records for training alignment; the legacy
physics-rate `cmd_log` is sampled after physics and can describe the incoming
command at a control boundary.

`terminal_time`, `terminal_q`, `terminal_v`, `terminal_prev_action`,
`terminal_phase_step`, `terminal_clock`, and `terminal_h_in` / `terminal_c_in`
retain the final state after physics. Use it as the final 20 ms transition
target only when `terminal_control_aligned` is true. A stop in the middle of a
control interval is retained and explicitly marked false. A rollout with no
policy queries has terminal fields only and supplies no training transitions.

The full-order archive also includes `policy_log_schema` (`g1_jog_policy_v1`),
`policy_log_convention`, `policy_kind`, `policy_state_names`, policy/model SHA-256
hashes, joint names, and the gait period from ONNX metadata. Parallel logs add
the collection configuration in `generation_metadata`; single logs identify
`command_source` as `raw_twist` or `shaped_stick`. Large policy arrays are saved
once in the full-order file; ROM files keep their plotting data and timestamps.
Load all fields with `np.load(path, allow_pickle=False)`.

Install the dependencies in `environment.yml` and `pytest`, then check the
logger with `python -m pytest tests/test_jog_logging.py -q`. The tests use the
actual ONNX policy and CPU MuJoCo to check exact policy replay, zero-command
masking, free-running clocks, resets, batch memory ordering, partial intervals,
and physics logging rates from 25 to 500 Hz. The parallel-loop tests substitute
CPU MuJoCo for the GPU stepper; an actual MuJoCo Warp GPU smoke run still
requires CUDA.

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

### Full jogging KCF collection

`sim/parallel/jog_collection_v1.json` defines a reproducible collection of 10,240
20-second episodes at 50 Hz. Run it on GPU 0 with:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_jog_dataset.py
```

The runner first checks a separate 256-episode pilot, then collects 20 batches
of 512 episodes. The eight profile families cover standing, sustained motion,
speed ramps within the trained bands, curves, pivots, settled starts/stops,
short stops with early restarts, and directed transitions between all seven
command behaviors (standing; forward/backward straight motion; left/right
curves; left/right pivots). Moving episodes finish with at least three seconds
of zero command. Commands retain the deployed controller's discontinuities;
neither the LSTM nor its clock is reset when a command changes.

Initial joint positions vary by up to 0.01 rad, and world origins/yaws vary
without changing the robot's body-frame task. Initial clock indices include
deployment-like zero-phase starts and all 43 gait phases. Physics, policy weights,
PD gains, and trained twist bands are the same as the simulator.

Compressed full-order archives, per-batch checksums/audits, and progress manifests
are under `data/data/jog_kcf_collection_v1/`. Use the `full/` directory as the
raw training source; `pilot/` is a separate collection check. Raw positions and
base linear velocities remain in world coordinates, with wxyz orientations and
body angular velocities. The KCF processing pipeline on `integrated-pose-runtime`
owns whole-trajectory splitting and the fixed ground/yaw frame convention in
`utils/relative_frame.py`. The collector assigns stable trajectory IDs and profile
labels, and records the frame helper's hash, without assigning dataset splits.
That branch's `data.py` / `process.py` pipeline assembles the 359-feature jogging
state from the 29-joint layout, recorded LSTM hidden/cell memory, previous
actions, and ungated gait clock. The selected 100-step jogging checkpoint and
its validation/test plots are published in
`KCF_MPC/results/g1_29dof_h100_v1/`; the checkpoint uses Git LFS.

Every batch checks exact command/state alignment, phase continuity, LSTM memory
continuity, sampled ONNX replay, and relative-frame round trips. Raw failures are
retained. `valid_transition_mask` marks the valid prefix before the existing
0.5 m height / 0.8 rad tilt envelope is crossed, and `first_invalid_transition`
identifies the boundary. The final audit requires all profiles, starts, stops,
early restarts, long zero holds, gait phases, and at least 50 valid examples of
each of the 42 directed behavior transitions. These are collection checks;
held-out KCF accuracy still needs to be established during model training.

Re-running the command resumes completed batches after checksum verification.
It rejects changed settings or collection code rather than mixing incompatible
recordings. Collection status is in `run_status.json`, with detailed totals in
`full/manifest.json`. No training process is launched by this collector.
Use `python scripts/jog_collection_status.py` to inspect progress without
initializing CUDA or reading the large raw arrays.
After completion, `python scripts/report_jog_collection.py` writes a coverage
figure, command-versus-measured motion examples for all eight profiles, and a
JSON summary under `data/data/jog_kcf_collection_v1/reports/`. That directory
also contains `array_schema.json`, listing each saved field's shape and dtype.

### Smooth paths for jogging KCF/MPC

The separate smooth-input supplement adds 1,536 one-minute episodes at 50 Hz,
after a 96-episode pilot. Its seven profiles are figure-eights (25%), circles
and ovals (15%), sinusoidal slaloms (15%), independent speed/yaw multisine (15%),
quintic interpolation between speed/yaw targets (15%), smooth pivot-rate changes
(5%), and smooth driving segments interleaved with stops/restarts (10%).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_smooth_jog_dataset.py --phase pilot
python scripts/report_smooth_jog_collection.py --phase pilot
CUDA_VISIBLE_DEVICES=0 python scripts/collect_smooth_jog_dataset.py --phase full
python scripts/report_smooth_jog_collection.py
```

`sim/parallel/jog_smooth_collection_v1.json` defines the size and seed. Data,
checksums, manifests, and plots are under `data/data/jog_kcf_smooth_collection_v1/`.
Inspect progress with `python scripts/jog_collection_status.py --directory
data/data/jog_kcf_smooth_collection_v1`. The collector reuses the same physics,
policy, logging, validity envelope, and relative-frame helper as the first
collection. It gives the supplement a distinct collection ID and episode-ID
range. Whole-trajectory splitting remains in KCF processing.

Both forward and backward path traversals are included. Figure-eights and ovals
use arc-length parameterization and select spatial scale/traversal period to
respect the policy's speed and turn-rate bands. Every periodic-path episode
commands at least one full reference cycle. Within driving modes both speed and yaw rate
vary smoothly, including small turn rates near zero. Startup/shutdown keep the
deployed mode switch: driving commands jump between zero and the supported
speed band, and pivots stay within their separate yaw-rate band.

`command_integrated_reference_log` is an ideal unicycle path in the initial
ground/yaw frame. It is a reference derived from the input sequence; the robot
is simulated freely and its actual motion is logged independently. Per-episode
parameters, core-cycle indices, nominal loop closure, and measured loop closure
are saved. Quality checks require complete valid reference cycles, smooth
variation in both command channels, and the original controller-state checks.
For periodic paths, the variant suffix identifies the signed path
parameterization. Randomized arc offsets determine the initial turn; read its
actual sign from `policy_cmd_log` rather than inferring it from the suffix.

The report writes `smooth_paths.png` (reference versus measured paths),
`smooth_commands.png` (commands versus measured velocities), checksum results,
and actual loop-error statistics. After full completion it also creates
`combined_corpus.json`, referencing the two full raw collections and excluding
pilots. Use that combined corpus when adding the 29-joint/LSTM KCF adapter;
the smooth and step/hold recordings complement each other.

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

For a terminal without a desktop display, save the jogging position and velocity
plots as PNG images with `python data/plot_29dof_data.py --save`. This writes
`positions.png` and `velocities.png` beside the dataset in a `<dataset>_plots/`
directory and exits. Use `--save <directory>` to choose another destination.

To inspect the actual robot motion without an X11 viewer, run
`python sim/single/preview_jog.py --gpu 0`. This uses the single-robot MuJoCo
simulation and ONNX policy, with EGL rendering on GPU 0 and `ffmpeg` encoding.
It writes six MP4 demonstrations and a combined `all_motions.mp4` under
`data/data/policy_preview/`, with commanded/measured velocity overlays and JSON
summaries. The cases cover standing, forward speed changes, backward motion,
curves, pivots, and stopping/restarting. Each case begins from a fresh simulation;
the policy memory and clock persist across command changes within that case.
Use `--preset stop_restart` (or another case name) to render one demonstration.

Reduced-order — Matplotlib plots of `q_rom` / `v_rom` (with body-frame cmd
overlay) plus a looping 2D top-down animation with heading triangles:
```bash
python data/plot_rom_data.py --walk              # walk GPU dataset _01 (--dataset N for another)
python data/plot_rom_data.py lissajous --walk    # the walk run saved with --log lissajous
python data/plot_rom_data.py --jog               # jog GPU dataset _01 (--dataset N for another)
python data/plot_rom_data.py myjog --jog         # a simulation_jog.py run
```
