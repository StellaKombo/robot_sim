# robot_sim

Run and compare KCF-MPC and single-integrator MPC on a G1 humanoid in MuJoCo,
using the same frozen RL locomotion policy. The selected model is the original
epoch-1540 checkpoint; no fine-tuned candidate was accepted.

This checkout is a simulation/testing handover, not a hardware driver or a
hardware-qualified implementation. The active optimizer is GPU-resident iCEM;
see the [formulation and usage](KCF_MPC/README.md).

## Install

Install Conda, `make`, Git LFS and a C++17-capable `g++`, then clone with the recorded submodule:

```bash
git clone --recurse-submodules <this repo's URL>
cd robot_sim
make install
conda activate robot_sim
make check
```

`make install` retrieves the recorded submodule revision and selected LFS model,
creates the environment in [environment.yml](environment.yml), and sets
`ROBOT_SIM_ROOT_DIR` for that environment. A detached submodule HEAD is normal.
Do not update the submodule to an arbitrary remote tip. No training data is needed.
The default JAX backend needs a CUDA-12-compatible NVIDIA driver and GPU access.
The fixed KCF CUDA kernels compile once at startup using the CUDA tools bundled
with the Python environment and `g++`. Unsupported GPU occupancy or compilation
fails explicitly; there is no silent fallback to the slower GPU implementation.
For an existing environment, use `make update` after dependency changes.

Regression tests are kept locally by maintainers, not shipped in this checkout.
If you have those local test folders, their optional dependencies and command are:

```bash
python -m pip install -r requirements-dev.txt
make test
```

Local tests use CPU JAX and non-interactive plots; GPU-only checks skip without CUDA.
The shared smoke/validation tools below do not need the regression-test folders.

## First run: KCF versus SI

Run these sequentially from the repository root, with fresh run names and no
other simulator, joystick, or command source running:

```bash
CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py smoke_01 \
  --smoke-test --synchronous --headless

CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py kcf_01 \
  --lissajous --initial-xy 0 0 --duration 20 --headless

CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py si_01 \
  --lissajous --initial-xy 0 0 --duration 20 --headless --single-integrator
```

Omit `--headless` to open the MuJoCo viewer. Synchronous mode pauses physics
during each MPC solve: 20 simulated seconds can take longer in wall time.
Trial viewers are observation-only: camera controls work, but GUI resets and
mouse forces cannot alter the measured experiment.
Omit `--synchronous` for asynchronous trials, including figure-eight testing.
The native GPU evaluator and forward-only iCEM are selected. Evaluate tracking,
completed worker/request timing, missed handovers and fallbacks together.
Finite MuJoCo runs do not establish hard-real-time or hardware qualification.
`--check` prints the complete plan without starting physics.

Both figure-eight runs use the same scenario, formulation and command bounds.
KCF uses the top-level gains in `KCF_MPC/config/mpc_config.py`; SI uses its
`single_integrator_gains` block (`state_weights`, `terminal_state_weights`,
`input_weight`, `command_change_weights`). SI has separately tuned gains.
Editing the SI block affects SI only, in both asynchronous and synchronous runs.
This is an independently tuned comparison, not a shared-gain comparison.
Neither configuration establishes general superiority.
The trial inherits the configured reference; `fixed` mode tests recovery with a
continuous hold requirement and timeouts. Only iCEM is supported.

For smooth waypoint tracking with explicit stops, select `reference_mode="waypoints"`
and edit `waypoint_reference` in `KCF_MPC/config/mpc_config.py`.
No trajectory flag or extra JSON file is needed. Add `--single-integrator` for
the same SI comparison. The configured example includes smooth startup and a
final hold (24 s total).
Current iCEM constrains command rates inside optimization and penalizes predicted
tilt. Startup references hold the measured pose briefly, then blend into motion.
These are transient-quality measures, not a guarantee of hardware balance.

`KCF_MPC/mpc/` is the controller library; `mpc/native/` contains its fixed CUDA
rollout and JAX FFI bridge. Simulation,
validation, replay and profiling entry points are in
[KCF_MPC/scripts/](KCF_MPC/scripts/README.md), not alternate controllers.

## Playback and plots

Trials save logs, plans and traces under `data/data/<run>/`. Overlay the two
recordings at one simulated second per wall-clock second:

```bash
python data/compare_rom_data.py kcf_01 si_01
python data/plot_policy_actions.py kcf_01 si_01 --start 0 --zoom 6
python data/plot_rom_data.py kcf_01
python data/plot_23dof_data.py kcf_01
```

The comparison overlays measured pose trails and heading triangles, with static
plots of position, heading sine/cosine, body velocity, commands and saved reference.
It replays logs; it does not rerun physics. Missing metadata or mismatched initial
poses/references produce warnings. Use `--once`, `--speed 0.5`, or
`--start 10 --end 15` to control playback. Single-run plotters require a run name.

Save comparison and RL diagnostics without a GUI:

```bash
MPLBACKEND=Agg python data/compare_rom_data.py kcf_01 si_01 \
  --no-playback --output-dir KCF_MPC/reports/comparison_01
MPLBACKEND=Agg python data/plot_policy_actions.py kcf_01 si_01 \
  --start 0 --output-dir KCF_MPC/reports/comparison_01/rl_actions --no-show
```

The ROM logger retains `[px, py, theta]`; plots derive sine/cosine without
rewriting recordings. RL diagnostics inspect the raw 23 policy actions and joint
targets, not just achieved base velocity. They align previous-action memory with
its actual policy output time and exclude artificial startup history. Gait peaks
and harmonics are expected; high-frequency content alone does not prove noise
or that MPC caused an oscillation. No trace smoothing is applied.

## Direct asynchronous simulation

For interactive testing, open two terminals in the same environment. Terminal 1:

```bash
python sim/single/simulation.py --log interactive_01 --log_hz 100
```

Terminal 2, choose one:

```bash
python sim/single/control_custom.py
python sim/single/control_custom.py --single-integrator
python sim/single/control_kb.py
python sim/single/control_joy.py
```

The simulator owns physics, RL inference, PD control, logging and the viewer.
MPC sends body-frame `[vx, vy, wz]` to the RL policy over localhost ZeroMQ
(ports 5555/5556). Policy ticks are 20 ms, physics ticks are 2 ms. Commands time
out to zero after 0.5 s without an update. Select `reference_mode` in
`KCF_MPC/config/mpc_config.py`: `fixed`, `lissajous`, `waypoints`, `rounded_square`,
`hairpin`, `slalom`, or `stop_turn_go`.
Targets, figure-eight dimensions/speed, and timed waypoint poses are defined
there; no trial script is needed. Both the standalone controller and trial launcher
accept `--reference-mode NAME` as a process-local override. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py kcf_hairpin_01 \
  --reference-mode hairpin --duration 120 --synchronous
```

Add `--single-integrator` for SI, or `--headless` to suppress the trial viewer.
The square, hairpin and slalom keep moving through smooth finite-radius turns;
stop–turn–go includes standing and in-place turns. Tune their named config blocks;
see [reference parameters](KCF_MPC/README.md#select-a-reference-in-the-controller-config).
`--lissajous` remains an optional process-local
override. The keyboard window must have focus; the
joystick entry point expects an Xbox-compatible controller.

Stop the simulator cleanly to save its logs. Reusing an interactive log name
overwrites it. `--log_hz` must divide the 500 Hz physics rate. Asynchronous timing
affects results; lockstep tests isolate solver timing but do not qualify real-time
deployment. Do not run multiple command sources together.

## Code map and recovery

- `models/`, `policy/`: robot XML/meshes, frozen RL weights and policy configuration.
- `sim/single/`: simulator and command-source entry points.
- `utils/kcf_controller.py`, `utils/kcf_reference.py`, `utils/g1_kcf_state*.py`:
  controller lifecycle, references, native-85 encoding and fixed-frame adapter.
- `KCF_MPC/config/mpc_config.py`: selected checkpoint and numerical settings.
- `KCF_MPC/models/`, `KCF_MPC/mpc/`: learned-model inference and MPC library.
- `KCF_MPC/scripts/`, `data/*.py`: simulation runs, validation and diagnostics.
