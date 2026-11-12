# robot_sim environment setup
ENV := robot_sim

.PHONY: install update uninstall test check

# fetch the KCF_MPC submodule (its model weights need git-lfs), create the conda env, and set
# ROBOT_SIM_ROOT_DIR to this repo
install:
	git lfs install --local
	git submodule update --init --recursive
	git -C KCF_MPC lfs install --local
	git -C KCF_MPC lfs pull --include="results/g1_kcf_handoff_epoch01540/*.pt" --exclude=""
	conda env create -f environment.yml
	conda env config vars set -n $(ENV) ROBOT_SIM_ROOT_DIR="$(CURDIR)"

# update the conda env from environment.yml and refresh the root path
update:
	conda env update -n $(ENV) -f environment.yml --prune
	conda env config vars set -n $(ENV) ROBOT_SIM_ROOT_DIR="$(CURDIR)"

# Maintainer-only: regression suites stay local, not in a shared checkout.
test:
	@if [ ! -d tests ] || [ ! -d KCF_MPC/tests ]; then \
		echo "Regression tests are local-only; use make check and the MuJoCo smoke tools in a shared checkout."; \
		exit 2; \
	fi
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest -q

# Inspect the configured controller without starting physics or opening sockets.
check:
	python KCF_MPC/scripts/visual_recovery_trial.py handover_check --synchronous --headless --check

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
