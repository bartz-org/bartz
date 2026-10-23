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

# Refuse -j: recipes manage their own parallelism (pytest-xdist, asv) and the
# release pipeline relies on serial prerequisite order (e.g. build before
# check-dist). Global because GNU make <4.4 can't scope .NOTPARALLEL to a target.
.NOTPARALLEL:

# define command to run python
CUDA_VERSION = $(shell nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | cut -d' ' -f3)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN = uv run --dev $(EXTRAS)

# Anchor all date-based dependency and release policies to the start of the
# current UTC day, as an explicit RFC 3339 instant so the config scripts resolve
# the same absolute cutoff regardless of the runner's local timezone. Repeated
# `make release` runs on the same UTC day resolve identically; crossing into the
# next UTC day may introduce one fresh bump. `:=` samples the date once per make
# invocation.
TODAY := $(shell date -u +%Y-%m-%dT00:00:00Z)

# Dependency cooldown in days, read from `[update] cooldown_days` in prek.toml,
# where it also applies to a bare `prek update`. `update-python-deps` pins
# pyproject's `[tool.uv] exclude-newer` to TODAY minus this many days, so
# releases skip just-published versions. It is pinned to an absolute instant
# (not uv's relative `1 week` span) so the release lock and the uv-lock hook
# resolve the same cutoff and don't churn uv.lock. Lazy (`=`), so only the
# targets that use it pay for reading it; tomllib needs python>=3.11.
COOLDOWN_DAYS = $(or $(shell uv run --no-project --python='>=3.11' python -c 'import tomllib; print(tomllib.load(open("prek.toml", "rb"))["update"]["cooldown_days"])'),$(error cannot read [update] cooldown_days from prek.toml))

# define command to run python with oldest supported dependencies
# OLD_DATE / OLD_DELAY_DAYS / BUMP_PYTHON_VERSION_DATE / NUM_SUPPORTED_PYTHON_RELEASES
# drive the `update-oldest-deps` policy.
OLD_DATE = 2025-10-04T00:00:00Z
OLD_DELAY_DAYS = 365
BUMP_PYTHON_VERSION_DATE = 10-31
NUM_SUPPORTED_PYTHON_RELEASES = 5
OLD_PYTHON = $(shell grep 'requires-python' pyproject.toml | sed 's/.*>=\([0-9.]*\).*/\1/')
# WORKAROUND(stochtree<=0.4.2): 0.4.2 is the oldest stochtree whose probit output
# matches bartz, but it is newer than OLD_DATE, so the old toolchain cannot
# resolve it. Exempt stochtree from the cutoff; drop --exclude-newer-package once
# the stochtree floor rises above 0.4.2 (check-workarounds will flag this then).
UV_RUN_OLD = $(UV_RUN) --python=$(OLD_PYTHON) --resolution=lowest-direct --exclude-newer=$(OLD_DATE) --exclude-newer-package="stochtree=0 days" --isolated

.PHONY: help
help:
	@echo "Available targets:"
	@echo "- setup: create the Python environment for development"
	@echo "- tests: run unit tests on cpu, saving coverage information"
	@echo "- tests-single-cpu: like \`tests\` but with a single jax cpu device"
	@echo "- tests-old: run unit tests on cpu with oldest supported python and dependencies"
	@echo '- tests-gpu: like `tests` but on gpu'
	@echo '- tests-gpu-old: like `tests-old` but on gpu'
	@echo "- docs: build html documentation"
	@echo "- docs-latest: build html documentation for latest release"
	@echo "- covreport: build html coverage report"
	@echo "- covcheck: check coverage is above some thresholds"
	@echo "- diffcov: check changed-lines coverage vs DIFF_BASE (default origin/main)"
	@echo "- update-deps: update-python-deps + update-other-deps"
	@echo "- update-python-deps: pin the dep cooldown, then upgrade uv.lock to the latest allowed deps"
	@echo "- update-other-deps: upgrade prek hooks and GitHub Actions (not date-pinned)"
	@echo "- update-oldest-deps: advance OLD_DATE and refresh oldest-supported pins in pyproject.toml"
	@echo "- check-committed: verify there are no uncommitted changes"
	@echo "- check-changelog: verify the topmost changelog section is dated today"
	@echo "- check-npz-frozen: verify the test archives of the file format are frozen"
	@echo "- build: build the python wheel and sdist"
	@echo "- check-dist: verify dist/ artifacts carry the release version"
	@echo "- release: run tests, build, and upload to PyPI (run on main)"
	@echo "- version-tag: create local git tag for the topmost changelog version"
	@echo "- push-tag: push the version tag to origin"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- gh-release: publish GitHub release from docs/changelog.md"
	@echo "- asv-machine: initialize ~/.asv-machine.json with a human-readable id"
	@echo "- asv-run: run benchmarks on all unbenchmarked tagged releases and main"
	@echo "- asv-publish: create html benchmark report"
	@echo "- asv-preview: create html report and start server"
	@echo "- asv-main: run benchmarks on main branch"
	@echo "- asv-quick: run quick benchmarks on current code, no saving"
	@echo "- ipython: start an ipython shell with stuff pre-imported"
	@echo "- ipython-old: start an ipython shell with oldest supported python and dependencies"
	@echo "- lint: run prek hooks on all files"
	@echo
	@echo "Release workflow:"
	@echo "- describe release in docs/changelog.md (its topmost header sets the version, follow effver https://jacobtomlinson.dev/effver)"
	@echo "- $$ make setup"
	@echo "- $$ make update-deps"
	@echo "- $$ make release, will not release but runs all tests, iterate and debug"
	@echo '- run `make tests-gpu` on a gpu'
	@echo "- merge a PR with the changes and fixes"
	@echo "- re-run benchmarks on cpu & gpu, fix performance regressions"
	@echo '- open the README colab link, run `%pip install git+https://github.com/bartz-org/bartz@main` in a scratch cell, then run all, fix problems'
	@echo '- save notebook locally and commit it, merge PR'
	@echo "- on main: $$ make release"
	@echo "- merge fix PR and try again until make release passes"
	@echo "- if the online docs are not up-to-date, merge another PR to trigger a new merge CI"


################# SETUP #################

.PHONY: setup
setup:
	$(UV_RUN) prek install --prepare-hooks
	$(if $(filter 12 13,$(CUDA_VERSION)),JAX_PLATFORMS=cuda) $(UV_RUN) python -c 'import jax; jax.numpy.empty(0)'

.PHONY: lint
lint:
	$(UV_RUN) prek run $(if $(ARGS),$(ARGS),--all-files)

.PHONY: clean
clean:
	rm -fr .venv
	rm -fr dist
	rm -fr config/jax_cache
	rm -fr docs/_build
	rm -fr docs/reference/_autogen
	rm -fr .coverage* coverage.xml diffcov.md

################# TESTS #################

# Test groups: each is a chunk of pytest args (paths/nodeids + -k expression)
# that selects a balanced slice of the suite. CI runs one group per matrix cell
# (NPROC=0, no xdist), so total wall time is the slowest cell: the groups are
# balanced to land each cell around ~21 min on the slowest target. To run a
# single group locally (composes with any tests target):
#   make tests             GROUP=iface-v4
#   make tests-single-cpu  GROUP=iface-v7
#   make tests-old         GROUP=iface-v23
# Leaving GROUP unset runs the whole suite. The matrix in
# `.github/workflows/tests.yml` lists these same names; keep them in sync.
#
# Cost is dominated by test_interface.py (~4500s on tests-old): each bkw variant
# re-fits a class-scoped CachedBart fixture and recompiles the same functions, so
# each variant stays within one group, and the other suites fill the leftover
# budget as whole files or fixture-free slices (mcmcstep's Geweke test and
# TestMultichain). iface-v1 is the catch-all: besides its variant, it runs every
# file other than test_interface.py not in GROUPED_FILES, so a new test file is
# run there rather than never. Rough cost per group (CI wall seconds,
# tests/tests-single-cpu/tests-old; re-measure after big changes):
#   iface-v1 950/750/1250    iface-v23 1000/650/1250  iface-v4 1050/700/1200
#   iface-v5 1050/700/1250   iface-v6  1000/1100/1200 iface-v7 1050/1150/1250
GROUPED_FILES := tests/test_mcmcstep.py tests/test_jaxext.py tests/test_bcf.py tests/test_stochtree.py tests/test_debug.py tests/test_dgp.py
GROUP_iface-v1  := tests $(addprefix --ignore=,$(GROUPED_FILES)) -k "v1 or not test_interface.py"
GROUP_iface-v23 := tests/test_interface.py tests/test_jaxext.py 'tests/test_mcmcstep.py::TestMultichain' -k "v2 or v3 or not test_interface.py"
GROUP_iface-v4  := tests/test_interface.py -k v4
GROUP_iface-v5  := tests/test_interface.py tests/test_bcf.py tests/test_stochtree.py -k "v5 or not test_interface.py"
GROUP_iface-v6  := tests/test_interface.py tests/test_mcmcstep.py tests/test_debug.py tests/test_dgp.py -k "v6 or not (test_interface.py or test_joint_distribution or TestMultichain)"
GROUP_iface-v7  := tests/test_interface.py 'tests/test_mcmcstep.py::test_joint_distribution' -k "v7 or not test_interface.py or not (v1 or v2 or v3 or v4 or v5 or v6)"

GROUPS := iface-v1 iface-v23 iface-v4 iface-v5 iface-v6 iface-v7

SELECT = $(if $(GROUP),$(GROUP_$(GROUP)))

# Number of xdist workers. Default to 2 for local speed; CI overrides to 0
# (xdist off) because the small runners OOM under parallel test execution.
NPROC ?= 2

# On GPU, parallel workers overlap compilation (which dominates the run time),
# but the processes share the GPU memory and OOM on small cards, so turn off
# xdist when the GPU has less total memory than this (MiB). An explicit NPROC
# (command line or environment) takes precedence over the detection.
GPU_MIN_MEM = 15000
GPU_MEM = $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
ifeq ($(origin NPROC),file)
GPU_NPROC = $(shell [ "$(GPU_MEM)" -ge $(GPU_MIN_MEM) ] 2>/dev/null && echo $(NPROC) || echo 0)
else
GPU_NPROC = $(NPROC)
endif

TESTS_VARS = COVERAGE_FILE=.coverage.$@$(if $(GROUP),-$(GROUP))
TESTS_COMMAND = python -m pytest --cov --dist=worksteal --durations=1000
TESTS_CPU_VARS = $(TESTS_VARS) JAX_PLATFORMS=cpu
TESTS_CPU_COMMAND = $(TESTS_COMMAND) --platform=cpu --numprocesses=$(NPROC) $(SELECT)
# WORKAROUND(jax<0.10.3): jax 0.10.x exhausts/corrupts CUDA command buffers
# over a long test session (RESOURCE_EXHAUSTED / INTERNAL "Recorded commands
# are not empty"), so disable them for the GPU test run. Recheck whether this
# is still needed when raising the jax floor.
TESTS_GPU_VARS = $(TESTS_VARS) XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
TESTS_GPU_COMMAND = $(TESTS_COMMAND) --platform=gpu --numprocesses=$(GPU_NPROC) $(SELECT)

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
	# wipe autosummary stubs so removed symbols don't linger as stale pages
	rm -fr docs/reference/_autogen
	$(UV_RUN) make -C docs html
	test ! -d _site/docs-dev || rm -r _site/docs-dev
	mv docs/_build/html _site/docs-dev
	@echo
	@echo "Now open _site/index.html"

# Releases up to v0.13.0 have rpy2 in their dev deps: RPY2_CFFI_MODE=ABI lets
# it install without R. Drop it once the latest release no longer has rpy2.
.PHONY: docs-latest
docs-latest:
	@LATEST_TAG=$$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | sort -V | tail -1) && \
	if [ -z "$$LATEST_TAG" ]; then echo "No release tags found"; exit 1; fi && \
	echo "Building docs for $$LATEST_TAG" && \
	WORKTREE_DIR=$$(mktemp -d) && \
	trap "git worktree remove --force '$$WORKTREE_DIR' 2>/dev/null || rm -rf '$$WORKTREE_DIR'" EXIT && \
	git worktree add --detach "$$WORKTREE_DIR" "$$LATEST_TAG" && \
	RPY2_CFFI_MODE=ABI $(MAKE) -C "$$WORKTREE_DIR" docs && \
	test ! -d _site/docs || rm -r _site/docs && \
	mv "$$WORKTREE_DIR/_site/docs-dev" _site/docs
	@echo
	@echo "Now open _site/index.html"


################# COVERAGE #################

.PHONY: covreport
covreport:
	$(UV_RUN) coverage html --include='src/*'

.PHONY: covcheck
covcheck:
	$(UV_RUN) coverage report --include='tests/**/test_*.py'
	$(UV_RUN) coverage report --include='src/*'
	$(UV_RUN) coverage report --include='tests/**/test_*.py' --fail-under=99 --format=total
	$(UV_RUN) coverage report --include='src/*' --fail-under=90 --format=total

# Branch (changed-lines) coverage: fail if new/modified lines in src and tests
# are not covered above the threshold. DIFF_BASE is the ref to diff against;
# locally a feature branch is compared to origin/main. Writes a markdown report
# (used by CI to populate the job summary) and prints the text report.
DIFF_BASE ?= origin/main
DIFFCOV_FAIL_UNDER ?= 99
DIFFCOV_REPORT ?= diffcov.md

.PHONY: diffcov
diffcov:
	# -i: the xml is only an input to diff-cover, which assesses just the
	# changed files (always present in the checkout); never fail xml generation
	# over an unrelated path missing in the combined data.
	$(UV_RUN) coverage xml -i -o coverage.xml
	$(UV_RUN) diff-cover coverage.xml --compare-branch=$(DIFF_BASE) --fail-under=$(DIFFCOV_FAIL_UNDER) --format report:- --format markdown:$(DIFFCOV_REPORT)


################# DEPENDENCIES #################

# `update-deps` = latest python deps (uv) + everything else (prek hooks, GitHub
# Actions). Only `update-python-deps` runs inside `release`: it pins
# pyproject's exclude-newer to a fixed instant before locking, so repeated
# same-day release runs (and the uv-lock hook) resolve identically and don't
# churn. The others have no fixed-date knob, so they're bumped once by hand via
# `make update-deps` at the start of the release, outside the release loop.
PINACT = PINACT_MIN_AGE=$(COOLDOWN_DAYS) GITHUB_TOKEN=$${GITHUB_TOKEN:-$$(gh auth token)} $(UV_RUN) prek run pinact --stage manual --all-files
.PHONY: update-deps
update-deps: update-python-deps update-other-deps

.PHONY: update-python-deps
update-python-deps:
	$(UV_RUN) python -m config.update_cooldown --today=$(TODAY) --cooldown-days=$(COOLDOWN_DAYS)
	uv lock --upgrade

.PHONY: update-other-deps
update-other-deps:
	$(UV_RUN) prek update
	# Update GitHub Actions, skipping releases newer than the cooldown. prek
	# fails a hook that modifies files, so a second run checks it now passes.
	$(PINACT) || $(PINACT)

.PHONY: update-oldest-deps
update-oldest-deps:
	$(UV_RUN) python -m config.update_python_version --bump-date=$(BUMP_PYTHON_VERSION_DATE) --num-supported=$(NUM_SUPPORTED_PYTHON_RELEASES) --today=$(TODAY)
	$(UV_RUN) python -m config.update_oldest_deps --min-old-date=$(OLD_DATE) --delay-days=$(OLD_DELAY_DAYS) --today=$(TODAY)
	uv lock


################# RELEASE #################

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: check-changelog
check-changelog:
	$(UV_RUN) python -m config.util check_changelog --today=$(TODAY)

.PHONY: check-npz-frozen
check-npz-frozen:
	$(UV_RUN) python -m tests.test_npz --check-frozen

.PHONY: build
build:
	# remove stale artifacts: uv publish would upload everything in dist/
	rm -fr dist
	uv build

# The version is derived from the git tag at build time (hatch-vcs), so the
# tag must exist before `build` (`check-dist` verifies this on the
# artifacts). It is created locally first and pushed only after the build
# artifacts pass `check-dist` and `smoke-test`, to avoid editing a published
# tag if something fails in between.
.PHONY: release
release: check-changelog check-npz-frozen clean setup update-oldest-deps update-python-deps check-committed tests tests-single-cpu tests-old docs version-tag build upload gh-release
	@echo "Done!"

.PHONY: version-tag
version-tag: check-committed
	test $(shell git rev-parse --abbrev-ref HEAD) = main
	git fetch --tags
	$(eval VERSION_TAG := v$(shell $(UV_RUN) python -m config.util get_version))
	@if git rev-parse -q --verify refs/tags/$(VERSION_TAG) >/dev/null; then \
		test "$$(git rev-list -n 1 $(VERSION_TAG))" = "$$(git rev-parse HEAD)" \
			|| { echo "Tag $(VERSION_TAG) exists but points to a different commit;"; \
			     echo "if it is a leftover never pushed, delete it: git tag -d $(VERSION_TAG)"; exit 1; }; \
		echo "Tag $(VERSION_TAG) already exists on current commit"; \
	else \
		git tag --message=$(VERSION_TAG) $(VERSION_TAG); \
	fi

.PHONY: push-tag
push-tag: version-tag check-dist smoke-test
	git push origin $(VERSION_TAG)

# Untagged builds carry a +g<commit> local version segment, which PyPI and
# TestPyPI reject; this catches dist/ built before tagging, or gone stale.
.PHONY: check-dist
check-dist:
	@VERSION=$$($(UV_RUN) python -m config.util get_version) && \
	test -e "dist/bartz-$$VERSION.tar.gz" && test -e "dist/bartz-$$VERSION-py3-none-any.whl" || { \
		echo "dist/ does not carry the release version $$VERSION:"; \
		ls dist/ 2>/dev/null; \
		echo "build with the tag in place: make version-tag build"; \
		exit 1; }

.PHONY: smoke-test
smoke-test:
	uv run --isolated --no-project --with dist/*.whl python -c 'import bartz'
	uv run --isolated --no-project --with dist/*.tar.gz python -c 'import bartz'

.PHONY: upload
upload: push-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish
	@VERSION=$$($(UV_RUN) python -m config.util get_version) && \
	echo "Try to install bartz $$VERSION from PyPI" && \
	uv tool run --exclude-newer-package="bartz=0 days" --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

# Like `upload`, but the tag stays local: TestPyPI uploads are rehearsals.
.PHONY: upload-test
upload-test: version-tag check-dist smoke-test
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish --check-url=https://test.pypi.org/simple/ --publish-url=https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python -m config.util get_version) && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --exclude-newer-package="bartz=0 days" --index=https://test.pypi.org/simple/ --index-strategy=unsafe-best-match --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: gh-release
gh-release: push-tag
	$(UV_RUN) python -m config.util gh_release --today=$(TODAY)


################# BENCHMARKS #################

ASV = $(UV_RUN) python -m asv

.PHONY: asv-machine
asv-machine:
	$(UV_RUN) python -m config.asv_machine

.PHONY: asv-run
asv-run: ASV_REFS = $(shell $(UV_RUN) python -m config.refs_for_asv)
asv-run: asv-machine
	$(ASV) run --durations=all --skip-existing-successful --show-stderr "$(ASV_REFS)" $(ARGS)

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
