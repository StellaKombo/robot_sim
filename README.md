# robot_sim

Run and compare KCF-MPC and single-integrator MPC on a G1 humanoid in MuJoCo,
using the same frozen RL locomotion policy. The selected model is the original
epoch-1540 checkpoint; no fine-tuned candidate was accepted.

This checkout is a simulation/testing handover, not a hardware driver or a
real-time-qualified implementation. See the [formulation and handover](KCF_MPC/docs/HANDOVER.md).

## Install

Install Conda, `make`, and Git LFS, then clone with the recorded submodule:

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
For an existing environment, use `make update` after dependency changes.

Optional regression tests:

```bash
python -m pip install -r requirements-dev.txt
make test
```

Tests use CPU JAX and non-interactive plots; GPU-only checks skip without CUDA.

## First run: KCF versus SI

Run these sequentially from the repository root, with fresh run names and no
other simulator, joystick, or command source running:

```bash
CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py smoke_01 \
  --smoke-test --synchronous --headless

CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py kcf_01 \
  --lissajous --duration 20 --synchronous --headless

CUDA_VISIBLE_DEVICES=0 python KCF_MPC/scripts/visual_recovery_trial.py si_01 \
  --lissajous --duration 20 --synchronous --headless --single-integrator
```

Omit `--headless` to open the MuJoCo viewer. Synchronous mode pauses physics
during each MPC solve: 20 simulated seconds can take longer in wall time.
The figure-eight trial requires lockstep. For asynchronous figure-eight testing,
use the two-process commands below; fixed-target trials can omit `--synchronous`.
`--check` prints the complete plan without starting physics.

Both figure-eight runs use the same scenario, cost and command bounds; SI changes
only the prediction model. This is a shared-formulation comparison, not a claim
that either controller is independently optimally tuned or generally superior.
Without `--lissajous`, the trial tests fixed-target recovery with a continuous
hold requirement and timeouts. The existing trial08 profile uses `gtol=0.003`;
the direct controller uses `gtol=0.0015`. Cleanup does not retune either.

The nine files in `KCF_MPC/mpc/` are the controller library. Simulation,
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
python sim/single/control_custom.py --lissajous
python sim/single/control_custom.py --lissajous --single-integrator
python sim/single/control_kb.py
python sim/single/control_joy.py
```

The simulator owns physics, RL inference, PD control, logging and the viewer.
MPC sends body-frame `[vx, vy, wz]` to the RL policy over localhost ZeroMQ
(ports 5555/5556). Policy ticks are 20 ms, physics ticks are 2 ms. Commands time
out to zero after 0.5 s without an update. Without `--lissajous`, the custom
controller holds a fixed target. The keyboard window must have focus; the
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
- `KCF_MPC/scripts/`, `data/*.py`, `tests/`: simulation tests, diagnostics and regression checks.

Offline training, preprocessing, dataset collection, model-evaluation CLIs, old
MPC paths and experimental fine-tuning tooling are outside this handover. Their
source is recoverable from `precleanup/finetuning-2026-11-12` in both repositories.
See the [fine-tuning conclusion](KCF_MPC/docs/finetuning_summary.md) and
[cleanup verification](KCF_MPC/docs/cleanup_verification.md).

Local datasets, recordings, rejected weights and caches are ignored, not erased.
Only the selected model's inference/provenance files are shipped. Never load an
untrusted checkpoint or share another user's writable JAX compilation cache.
