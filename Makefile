# robot_sim environment setup
ENV := robot_sim

.PHONY: install update uninstall

# fetch the KCF_MPC submodule (its model weights need git-lfs), create the conda env, and set
# ROBOT_SIM_ROOT_DIR to this repo
install:
	git submodule update --init --recursive
	conda env create -f environment.yml
	conda env config vars set -n $(ENV) ROBOT_SIM_ROOT_DIR="$(CURDIR)"

# update the conda env from environment.yml and refresh the root path
update:
	conda env update -n $(ENV) -f environment.yml --prune
	conda env config vars set -n $(ENV) ROBOT_SIM_ROOT_DIR="$(CURDIR)"

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
