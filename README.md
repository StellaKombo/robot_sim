# robot_sim

Run and compare KCF-MPC and single-integrator MPC on a G1 humanoid in MuJoCo,
using the same frozen RL locomotion policy. The selected model is the epoch-3240 checkpoint, trained with the frozen
29-DoF walking policy. The simulator uses its matching ONNX policy and MuJoCo scene
from `deploy_robot/policy/g1/` and `deploy_robot/models/g1/`.

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
with the Python environment and `g++`.
For an existing environment, use `make update` after dependency changes.

If you have those local test folders, their optional dependencies and command are:

```bash
python -m pip install -r requirements-dev.txt
make test
```

## Direct asynchronous simulation

For interactive testing, open two terminals in the same environment. Terminal 1:

```bash
python sim/single/simulation.py --log interactive_01 --log_hz 100
```

To inspect the maze with the robot standing, run `python sim/single/simulation.py --maze`
without a controller. To navigate, start `python sim/single/control_custom.py --reference-mode maze`
in terminal 2. Edit walls (full lengths in metres), `spawn_pose` and `goal_pose` in
`sim/single/maze.json`; A* uses that same layout and waits for measured arrival before each turn.
Planning clearance, footprint and reference speeds are in `maze_reference` in
`KCF_MPC/config/mpc_config.py`. For another layout, set its `layout_path` and pass the same JSON
to `simulation.py --maze PATH.json`. The timed trial launcher does not support maze navigation.

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
stop–turn–go includes standing and in-place turns.

## Code map and recovery

- `policy/`: walking policy loader and configuration; `deploy_robot/models/g1/` and
  `deploy_robot/policy/g1/`: matching 29-DoF XML/meshes and frozen ONNX policy.
- `sim/single/`: simulator and command-source entry points.
- `utils/kcf_controller.py`, `utils/kcf_reference.py`, `utils/g1_kcf_state*.py`:
  controller lifecycle, references, native-103 encoding and fixed-frame adapter.
- `KCF_MPC/config/mpc_config.py`: selected checkpoint and numerical settings.
- `KCF_MPC/models/`, `KCF_MPC/mpc/`: learned-model inference and MPC library.
- `KCF_MPC/scripts/`, `data/*.py`: simulation runs, validation and diagnostics.
