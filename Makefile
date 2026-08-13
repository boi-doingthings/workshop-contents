.PHONY: bootstrap prepare-dev prepare-workshop notebook archive-notebook test preflight

bootstrap:
	./scripts/bootstrap.sh

prepare-dev:
	./scripts/prepare_assets.sh DEV_SMOKE

prepare-workshop:
	./scripts/prepare_assets.sh WORKSHOP_B200

notebook:
	./scripts/launch_notebook.sh

archive-notebook:
	test -n "$(RUN_DIR)"
	./scripts/container.sh bash -lc 'source .venv/bin/activate && python scripts/archive_notebook.py --run-dir "$(RUN_DIR)"'

test:
	./scripts/container.sh bash -lc 'source .venv/bin/activate && pytest -q'

preflight:
	./scripts/container.sh bash -lc 'source .venv/bin/activate && python scripts/run_profile.py --profile DEV_SMOKE --stage preflight'
