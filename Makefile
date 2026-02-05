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

COVERAGE_SUFFIX =

# define command to run python
CUDA_VERSION = $(shell nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | cut -d' ' -f3)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN = uv run --dev $(EXTRAS)

# define command to run python with oldest supported dependencies
OLD_PYTHON = $(shell grep 'requires-python' pyproject.toml | sed 's/.*>=\([0-9.]*\).*/\1/')
UV_RUN_OLD = $(UV_RUN) --python=$(OLD_PYTHON) --resolution=lowest-direct --exclude-newer=2025-05-15 --isolated

.PHONY: all
all:
	@echo "Available targets:"
	@echo "- setup: create R and Python environments for development"
	@echo "- tests: run unit tests on cpu, saving coverage information"
	@echo "- tests-old: run unit tests on cpu with oldest supported python and dependencies"
	@echo '- tests-gpu: like `tests` but on gpu'
	@echo '- tests-gpu-old: like `tests-old` but on gpu'
	@echo "- docs: build html documentation"
	@echo "- docs-latest: build html documentation for latest release"
	@echo "- covreport: build html coverage report"
	@echo "- covcheck: check coverage is above some thresholds"
	@echo "- update-deps: remove .venv, upgrade uv.lock, update pre-commit hooks"
	@echo "- copy-version: sync version from pyproject.toml to _version.py"
	@echo "- check-committed: verify there are no uncommitted changes"
	@echo "- release: packages the python module, invokes tests and docs first"
	@echo "- version-tag: create and push git tag for current version"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- asv-run: run benchmarks on all unbenchmarked tagged releases and main"
	@echo "- asv-publish: create html benchmark report"
	@echo "- asv-preview: create html report and start server"
	@echo "- asv-main: run benchmarks on main branch"
	@echo "- asv-quick: run quick benchmarks on current code, no saving"
	@echo "- ipython: start an ipython shell with stuff pre-imported"
	@echo "- ipython-old: start an ipython shell with oldest supported python and dependencies"
	@echo
	@echo "Release workflow:"
	@echo "- create a new branch"
	@echo "- $$ uv version --bump major|minor|patch"
	@echo "- describe release in docs/changelog.md"
	@echo "- commit"
	@echo "- open a PR"
	@echo "- $$ make release (iterate to fix problems)"
	@echo "- if CI does not pass, debug and go back to make release"
	@echo "- merge PR"
	@echo "- if CI does not pass, debug and go back to make release"
	@echo "- $$ make upload"
	@echo "- publish github release (updates zenodo automatically)"
	@echo "- if the online docs are not up-to-date, merge another PR to trigger a new merge CI"


.PHONY: setup
setup:
	Rscript -e "renv::restore()"
	$(UV_RUN) pre-commit install --install-hooks


################# TESTS #################

TESTS_VARS = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX)
TESTS_COMMAND = python -m pytest --cov --cov-context=test --dist=worksteal --durations=1000
TESTS_CPU_VARS = $(TESTS_VARS) JAX_PLATFORMS=cpu
TESTS_CPU_COMMAND = $(TESTS_COMMAND) --platform=cpu --numprocesses=2
TESTS_GPU_VARS = $(TESTS_VARS) XLA_PYTHON_CLIENT_MEM_FRACTION=.20
TESTS_GPU_COMMAND = $(TESTS_COMMAND) --platform=gpu --numprocesses=3

.PHONY: tests
tests:
	$(TESTS_CPU_VARS) $(UV_RUN) $(TESTS_CPU_COMMAND) $(ARGS)

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
	uv sync --all-groups --directory "$$WORKTREE_DIR" && \
	$(MAKE) -C "$$WORKTREE_DIR" docs && \
	test ! -d _site/docs || rm -r _site/docs && \
	mv "$$WORKTREE_DIR/_site/docs-dev" _site/docs
	@echo
	@echo "Now open _site/index.html"

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


################# RELEASE #################

.PHONY: update-deps
update-deps:
	uv lock --upgrade
	$(UV_RUN) pre-commit autoupdate

.PHONY: copy-version
copy-version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	$(UV_RUN) python config/util.py update_version

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: clean
clean:
	rm -fr .venv
	rm -fr dist
	rm -fr config/jax_cache

.PHONY: release
release: clean update-deps copy-version check-committed tests tests-old docs
	uv build

.PHONY: version-tag
version-tag: copy-version check-committed
	git fetch --tags
	git tag v$(shell uv run python -c 'import bartz; print(bartz.__version__)')
	git push --tags

.PHONY: upload
upload: version-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish
	@VERSION=$$(uv run python -c 'import bartz; print(bartz.__version__)') && \
	echo "Try to install bartz $$VERSION from PyPI" && \
	uv tool run --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: upload-test
upload-test: check-committed
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish --check-url=https://test.pypi.org/simple/ --publish-url=https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --index=https://test.pypi.org/simple/ --index-strategy=unsafe-best-match --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'


################# BENCHMARKS #################

ASV = $(UV_RUN) python -m asv

.PHONY: asv-run
asv-run:
	$(UV_RUN) python config/refs-for-asv.py | $(ASV) run --durations=all --skip-existing-successful --show-stderr HASHFILE:- $(ARGS)

.PHONY: asv-publish
asv-publish:
	$(ASV) publish $(ARGS)

.PHONY: asv-preview
asv-preview: asv-publish
	$(ASV) preview $(ARGS)

.PHONY: asv-main
asv-main:
	$(ASV) run --show-stderr main^! $(ARGS)

.PHONY: asv-quick
asv-quick:
	$(ASV) run --durations=all --python=same --quick --dry-run --show-stderr $(ARGS)


################# IPYTHON SHELL #################

.PHONY: ipython
ipython:
	IPYTHONDIR=config/ipython $(UV_RUN) python -m IPython $(ARGS)

.PHONY: ipython-old
ipython-old:
	IPYTHONDIR=config/ipython $(UV_RUN_OLD) python -m IPython $(ARGS)
