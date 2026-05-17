# bartz/Makefile
#
# Copyright (c) 2024-2026, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Makefile for running tests, prepare and upload a release.

# define command to run python
CUDA_VERSION = $(shell nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | cut -d' ' -f3)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN = uv run --dev $(EXTRAS)

# define command to run python with oldest supported dependencies
# OLD_DATE / OLD_DELAY_DAYS / BUMP_PYTHON_VERSION_DATE / NUM_SUPPORTED_PYTHON_RELEASES
# drive the `update-oldest-deps` policy.
OLD_DATE = 2025-05-15
OLD_DELAY_DAYS = 365
BUMP_PYTHON_VERSION_DATE = 10-31
NUM_SUPPORTED_PYTHON_RELEASES = 5
OLD_PYTHON = $(shell grep 'requires-python' pyproject.toml | sed 's/.*>=\([0-9.]*\).*/\1/')
UV_RUN_OLD = $(UV_RUN) --python=$(OLD_PYTHON) --resolution=lowest-direct --exclude-newer=$(OLD_DATE) --isolated

.PHONY: help
help:
	@echo "Available targets:"
	@echo "- setup: create R and Python environments for development"
	@echo "- tests: run unit tests on cpu, saving coverage information"
	@echo "- tests-single-cpu: like \`tests\` but with a single jax cpu device"
	@echo "- tests-old: run unit tests on cpu with oldest supported python and dependencies"
	@echo '- tests-gpu: like `tests` but on gpu'
	@echo '- tests-gpu-old: like `tests-old` but on gpu'
	@echo "- docs: build html documentation"
	@echo "- docs-latest: build html documentation for latest release"
	@echo "- covreport: build html coverage report"
	@echo "- covcheck: check coverage is above some thresholds"
	@echo "- update-deps: remove .venv, upgrade uv.lock, update pre-commit hooks"
	@echo "- update-oldest-deps: advance OLD_DATE and refresh oldest-supported pins in pyproject.toml"
	@echo "- copy-version: sync version from pyproject.toml to _version.py"
	@echo "- check-committed: verify there are no uncommitted changes"
	@echo "- release: packages the python module, invokes tests and docs first"
	@echo "- version-tag: create and push git tag for current version"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- asv-machine: initialize ~/.asv-machine.json with a human-readable id"
	@echo "- asv-run: run benchmarks on all unbenchmarked tagged releases and main"
	@echo "- asv-publish: create html benchmark report"
	@echo "- asv-preview: create html report and start server"
	@echo "- asv-main: run benchmarks on main branch"
	@echo "- asv-quick: run quick benchmarks on current code, no saving"
	@echo "- ipython: start an ipython shell with stuff pre-imported"
	@echo "- ipython-old: start an ipython shell with oldest supported python and dependencies"
	@echo "- lint: run pre-commit hooks on all files"
	@echo
	@echo "Update dependencies workflow:"
	@echo "- new PR"
	@echo "- $$ make update-deps"
	@echo "- $$ make tests  # and debug"
	@echo "- $$ make update-oldest-deps"
	@echo "- $$ make tests-old  # and debug"
	@echo
	@echo "Release workflow:"
	@echo "- do a PR that re-runs benchmarks"
	@echo "- create a new branch"
	@echo "- $$ uv version --bump major|minor|patch"
	@echo "- describe release in docs/changelog.md"
	@echo "- commit"
	@echo "- open a PR"
	@echo "- $$ make release (iterate to fix problems)"
	@echo "- if CI does not pass, debug and go back to make release"
	@echo "- merge PR"
	@echo "- if CI does not pass, debug and go back to open PR"
	@echo "- $$ make upload"
	@echo "- publish github release (updates zenodo automatically)"
	@echo "- if the online docs are not up-to-date, merge another PR to trigger a new merge CI"


################# SETUP #################

.PHONY: setup
setup:
	Rscript -e "renv::restore()"
	$(UV_RUN) pre-commit install --install-hooks

.PHONY: lint
lint:
	$(UV_RUN) pre-commit run $(if $(ARGS),$(ARGS),--all-files)

.PHONY: clean
clean:
	rm -fr .venv
	rm -fr dist
	rm -fr config/jax_cache
	rm -fr docs/_build

################# TESTS #################

# Test groups: each is a chunk of pytest args (paths/nodeids + -k expression)
# that selects a balanced slice of the suite. CI runs one group per matrix cell
# to fit each job under ~15 minutes on the slow `tests-old` target without
# xdist. To run a single group locally (composes with any tests target):
#   make tests             GROUP=v4light
#   make tests-single-cpu  GROUP=misc
#   make tests-old         GROUP=v1v7
# Leaving GROUP unset runs the whole suite.
#
# The two test files with a variant fixture (test_BART.py for v1/v2/v3 and
# test_interface.py for v4/v5/v6/v7) dominate cost: each variant re-fits the
# CachedBart and re-triggers JIT compilations. v4 and v5 are too heavy to fit
# in one group on their own, so they're sliced further: v4's test_equiv_sharding
# is carved out (no class-scoped fixture), and v5's TestWithCachedBart class is
# pulled into its own group (the cachedbart fixture, scope='class', is the bulk
# of the cost — splitting members across groups would pay it twice).
GROUP_misc        := tests/test_mcmcstep.py tests/test_mcmcloop.py tests/test_dgp.py tests/test_prepcovars.py tests/test_debug.py tests/test_meta.py 'tests/test_interface.py::test_equiv_sharding[v4]'
GROUP_v1v7        := tests/test_BART.py tests/test_interface.py -k "v1 or v7 or not (v2 or v3 or v4 or v5 or v6)"
GROUP_v2v3jaxext  := tests/test_BART.py tests/test_jaxext.py -k "v2 or v3 or jaxext"
GROUP_v5heavy     := tests/test_interface.py::TestWithCachedBart -k v5
GROUP_v4light     := tests/test_interface.py -k "v4 and not test_equiv_sharding"
GROUP_v5light-v6  := tests/test_interface.py -k "(v5 and not TestWithCachedBart) or v6"

GROUPS := misc v1v7 v2v3jaxext v5heavy v4light v5light-v6

SELECT = $(if $(GROUP),$(GROUP_$(GROUP)))

# Number of xdist workers. Default to 2 for local speed; CI overrides to 0
# (xdist off) because the small runners OOM under parallel test execution.
NPROC ?= 2

TESTS_VARS = COVERAGE_FILE=.coverage.$@$(if $(GROUP),-$(GROUP))
TESTS_COMMAND = python -m pytest --cov --cov-context=test --dist=worksteal --durations=1000
TESTS_CPU_VARS = $(TESTS_VARS) JAX_PLATFORMS=cpu
TESTS_CPU_COMMAND = $(TESTS_COMMAND) --platform=cpu --numprocesses=$(NPROC) $(SELECT)
TESTS_GPU_VARS = $(TESTS_VARS) XLA_PYTHON_CLIENT_PREALLOCATE=false
TESTS_GPU_COMMAND = $(TESTS_COMMAND) --platform=gpu --numprocesses=$(NPROC) $(SELECT)

.PHONY: tests
tests:
	$(TESTS_CPU_VARS) $(UV_RUN) $(TESTS_CPU_COMMAND) $(ARGS)

.PHONY: tests-single-cpu
tests-single-cpu:
	$(TESTS_CPU_VARS) $(UV_RUN) $(TESTS_CPU_COMMAND) --num-cpu-devices=1 $(ARGS)

.PHONY: tests-old
tests-old:
	$(TESTS_CPU_VARS) $(UV_RUN_OLD) $(TESTS_CPU_COMMAND) $(ARGS)

.PHONY: tests-gpu
tests-gpu:
	nvidia-smi
	$(TESTS_GPU_VARS) $(UV_RUN) $(TESTS_GPU_COMMAND) $(ARGS)

.PHONY: tests-gpu-old
tests-gpu-old:
	nvidia-smi
	$(TESTS_GPU_VARS) $(UV_RUN_OLD) $(TESTS_GPU_COMMAND) $(ARGS)


################# DOCS #################

.PHONY: docs
docs:
	$(UV_RUN) make -C docs html
	test ! -d _site/docs-dev || rm -r _site/docs-dev
	mv docs/_build/html _site/docs-dev
	@echo
	@echo "Now open _site/index.html"

.PHONY: docs-latest
docs-latest:
	@LATEST_TAG=$$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | sort -V | tail -1) && \
	if [ -z "$$LATEST_TAG" ]; then echo "No release tags found"; exit 1; fi && \
	echo "Building docs for $$LATEST_TAG" && \
	WORKTREE_DIR=$$(mktemp -d) && \
	trap "git worktree remove --force '$$WORKTREE_DIR' 2>/dev/null || rm -rf '$$WORKTREE_DIR'" EXIT && \
	git worktree add --detach "$$WORKTREE_DIR" "$$LATEST_TAG" && \
	$(MAKE) -C "$$WORKTREE_DIR" docs && \
	test ! -d _site/docs || rm -r _site/docs && \
	mv "$$WORKTREE_DIR/_site/docs-dev" _site/docs
	@echo
	@echo "Now open _site/index.html"


################# COVERAGE #################

.PHONY: covreport
covreport:
	$(UV_RUN) coverage combine --keep
	$(UV_RUN) coverage html --include='src/*'

.PHONY: covcheck
covcheck:
	$(UV_RUN) coverage combine --keep
	$(UV_RUN) coverage report --include='tests/**/test_*.py'
	$(UV_RUN) coverage report --include='src/*'
	$(UV_RUN) coverage report --include='tests/**/test_*.py' --fail-under=99 --format=total
	$(UV_RUN) coverage report --include='src/*' --fail-under=90 --format=total


################# DEPENDENCIES #################

.PHONY: update-deps
update-deps:
	uv lock --upgrade
	$(UV_RUN) pre-commit autoupdate

.PHONY: update-oldest-deps
update-oldest-deps:
	$(UV_RUN) python config/update_python_version.py --bump-date=$(BUMP_PYTHON_VERSION_DATE) --num-supported=$(NUM_SUPPORTED_PYTHON_RELEASES)
	$(UV_RUN) python config/update_oldest_deps.py --min-old-date=$(OLD_DATE) --delay-days=$(OLD_DELAY_DAYS)
	uv lock


################# RELEASE #################

.PHONY: copy-version
copy-version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	$(UV_RUN) python config/util.py update_version

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: release
release: clean update-deps copy-version check-committed tests tests-single-cpu tests-old docs
	uv build

.PHONY: version-tag
version-tag: copy-version check-committed
	test $(shell git rev-parse --abbrev-ref HEAD) = main
	git fetch --tags
	$(eval VERSION_TAG := v$(shell uv run python -c 'import bartz; print(bartz.__version__)'))
	@if git rev-parse -q --verify refs/tags/$(VERSION_TAG) >/dev/null; then \
		test "$$(git rev-list -n 1 $(VERSION_TAG))" = "$$(git rev-parse HEAD)" \
			|| { echo "Tag $(VERSION_TAG) exists but points to a different commit"; exit 1; }; \
		echo "Tag $(VERSION_TAG) already exists on current commit"; \
	else \
		git tag --message=$(VERSION_TAG) $(VERSION_TAG); \
	fi
	git push origin $(VERSION_TAG)

.PHONY: smoke-test
smoke-test:
	uv run --isolated --no-project --with dist/*.whl python -c 'import bartz'
	uv run --isolated --no-project --with dist/*.tar.gz python -c 'import bartz'

.PHONY: upload
upload: smoke-test version-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish
	@VERSION=$$(uv run python -c 'import bartz; print(bartz.__version__)') && \
	echo "Try to install bartz $$VERSION from PyPI" && \
	uv tool run --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: upload-test
upload-test: smoke-test check-committed
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish --check-url=https://test.pypi.org/simple/ --publish-url=https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --index=https://test.pypi.org/simple/ --index-strategy=unsafe-best-match --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'


################# BENCHMARKS #################

ASV = $(UV_RUN) python -m asv

.PHONY: asv-machine
asv-machine:
	$(UV_RUN) python config/asv_machine.py

.PHONY: asv-run
asv-run: asv-machine
	$(UV_RUN) python config/refs-for-asv.py | $(ASV) run --durations=all --skip-existing-successful --show-stderr HASHFILE:- $(ARGS)

.PHONY: asv-publish
asv-publish:
	$(ASV) publish $(ARGS)

.PHONY: asv-preview
asv-preview: asv-publish
	$(ASV) preview $(ARGS)

.PHONY: asv-main
asv-main: asv-machine
	$(ASV) run --show-stderr main^! $(ARGS)

.PHONY: asv-quick
asv-quick: asv-machine
	$(ASV) run --durations=all --python=same --quick --dry-run --show-stderr $(ARGS)


################# IPYTHON SHELL #################

.PHONY: ipython
ipython:
	IPYTHONDIR=config/ipython $(UV_RUN) python -m IPython $(ARGS)

.PHONY: ipython-old
ipython-old:
	IPYTHONDIR=config/ipython $(UV_RUN_OLD) python -m IPython $(ARGS)
