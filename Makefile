# robot_sim environment setup
ENV := robot_sim

.PHONY: install update uninstall

# create the conda env from environment.yml
install:
	conda env create -f environment.yml

# update the conda env from environment.yml
update:
	conda env update -n $(ENV) -f environment.yml --prune

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
