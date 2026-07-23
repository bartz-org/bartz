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

# define command to run python with oldest supported dependencies
# OLD_DATE / OLD_DELAY_DAYS / BUMP_PYTHON_VERSION_DATE / NUM_SUPPORTED_PYTHON_RELEASES
# drive the `update-oldest-deps` policy.
OLD_DATE = 2025-07-21
OLD_DELAY_DAYS = 365
BUMP_PYTHON_VERSION_DATE = 10-31
NUM_SUPPORTED_PYTHON_RELEASES = 5
OLD_PYTHON = $(shell grep 'requires-python' pyproject.toml | sed 's/.*>=\([0-9.]*\).*/\1/')
# WORKAROUND(stochtree<=0.4.2): 0.4.2 is the oldest stochtree whose probit output
# matches bartz, but it is newer than OLD_DATE, so the old toolchain cannot
# resolve it. Exempt stochtree from the cutoff; drop --exclude-newer-package once
# the stochtree floor rises above 0.4.2 (check-workarounds will flag this then).
# WORKAROUND(rbartpackages<0.12.0): the first release (0.1.0) was uploaded on
# 2026-06-05, newer than OLD_DATE, so the old toolchain cannot resolve it.
# Exempt rbartpackages from the cutoff. The exemption is unneeded once OLD_DATE
# passes the upload date of the rbartpackages floor release; 0.12.0 is a guess
# at the floor around that time (~June 2027). On trigger, drop
# --exclude-newer-package if OLD_DATE has caught up, else bump the bound.
UV_RUN_OLD = $(UV_RUN) --python=$(OLD_PYTHON) --resolution=lowest-direct --exclude-newer=$(OLD_DATE) --exclude-newer-package="stochtree=0 days" --exclude-newer-package="rbartpackages=0 days" --isolated

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
	@echo "- diffcov: check changed-lines coverage vs DIFF_BASE (default origin/main)"
	@echo "- update-deps: upgrade uv.lock and renv.lock, update pre-commit hooks"
	@echo "- update-oldest-deps: advance OLD_DATE and refresh oldest-supported pins in pyproject.toml"
	@echo "- check-committed: verify there are no uncommitted changes"
	@echo "- check-changelog: verify the topmost changelog section is dated today"
	@echo "- build: build the python wheel and sdist"
	@echo "- check-dist: verify dist/ artifacts carry the release version"
	@echo "- release: run tests, build, and upload to PyPI (run on main)"
	@echo "- version-tag: create local git tag for the topmost changelog version"
	@echo "- push-tag: push the version tag to origin"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- gh-release: create draft GitHub release from docs/changelog.md"
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
	@echo "Release workflow:"
	@echo "- describe release in docs/changelog.md (its topmost header sets the version, follow effver https://jacobtomlinson.dev/effver)"
	@echo "- $$ make release, will not release but runs all tests, iterate and debug"
	@echo '- run `make tests-gpu` on a gpu'
	@echo "- merge a PR with the changes and fixes"
	@echo "- re-run benchmarks on cpu & gpu, fix performance regressions"
	@echo '- open the README colab link, run `%pip install git+https://github.com/bartz-org/bartz@main` in a scratch cell, then run all, fix problems'
	@echo '- save notebook locally and commit it, merge PR'
	@echo "- on main: $$ make release"
	@echo "- merge fix PR and try again until make release passes"
	@echo "- publish the draft github release created by make release (updates zenodo automatically)"
	@echo "- if the online docs are not up-to-date, merge another PR to trigger a new merge CI"


################# SETUP #################

.PHONY: setup
setup:
	Rscript -e "renv::restore()"
	$(UV_RUN) pre-commit install --install-hooks
	$(if $(filter 12 13,$(CUDA_VERSION)),JAX_PLATFORMS=cuda) $(UV_RUN) python -c 'import jax; jax.numpy.empty(0)'

.PHONY: lint
lint:
	$(UV_RUN) pre-commit run $(if $(ARGS),$(ARGS),--all-files)

.PHONY: clean
clean:
	rm -fr .venv
	rm -fr dist
	rm -fr config/jax_cache
	rm -fr docs/_build
	rm -fr docs/reference/_autogen
	rm -fr .coverage* coverage.xml diffcov.md
	# `renv::clean()` only removes locks/tempdirs/unused packages, not the
	# whole library, so wipe the gitignored renv subdirs by hand to mirror
	# `rm -fr .venv`.
	rm -fr renv/library renv/staging renv/local renv/cellar renv/lock renv/python renv/sandbox

################# TESTS #################

# Test groups: each is a chunk of pytest args (paths/nodeids + -k expression)
# that selects a balanced slice of the suite. CI runs one group per matrix cell
# (NPROC=0, no xdist), so total wall time is the slowest cell: the groups are
# balanced to land each cell around ~14 min on the slow `tests-old` target (the
# iface-v4 floor, see below). To run a single group locally (composes with any
# tests target):
#   make tests             GROUP=iface-v4
#   make tests-single-cpu  GROUP=misc
#   make tests-old         GROUP=bart-v1
# Leaving GROUP unset runs the whole suite. The matrix in
# `.github/workflows/tests.yml` lists these same names; keep them in sync.
#
# Cost is dominated by test_interface.py (variants v4-v7, ~2200s deduped on
# tests-old) and test_BART.py (variants v1-v3, ~1000s): each variant re-fits a
# class-scoped CachedBart fixture, so a given variant's tests are kept within one
# group (splitting them would pay the fixture twice). The heavy v4/v5 variants
# are sliced where it's free to do so: the test_equiv_sharding tests have no
# class fixture, so they ride along in cheaper groups, and v5's TestWithCachedBart
# class is isolated. The leftover budget in each group is topped up with the
# cheap whole-file suites (mcmcloop, stochtree, jaxext, ...); where a whole file
# would overshoot, a single fixture-free class is peeled off instead (mcmcstep's
# TestMultichain, ~265s, rides in bart-v1 to fill its slack). iface-v4 is the
# irreducible floor (~850s, almost all CachedBart setup that can't be split); the
# other groups are balanced just under it. Rough tests-old cost per group (wall
# seconds, ~= the deduped CI `--durations` table; re-measure after big changes):
#   misc ~755  iface-v5v7 ~730  iface-v6 ~760  iface-v4 ~850  bart-v1 ~725  bart-v23 ~670
GROUP_misc        := tests/test_mcmcstep.py tests/test_mcmcloop.py tests/test_dgp.py tests/test_prepcovars.py tests/test_debug.py tests/test_meta.py tests/test_naming.py tests/test_docs.py tests/test_workarounds.py 'tests/test_interface.py::test_equiv_sharding[v7]' -k "not TestMultichain"
GROUP_iface-v5v7  := tests/test_interface.py -k "(v5 and TestWithCachedBart) or (v7 and not test_equiv_sharding)"
GROUP_iface-v6    := tests/test_interface.py -k "v6 or (v5 and not TestWithCachedBart)"
GROUP_iface-v4    := tests/test_interface.py -k "(v4 and not test_equiv_sharding) or not (v2 or v3 or v4 or v5 or v6 or v7)"
GROUP_bart-v1     := tests/test_BART.py tests/test_jaxext.py tests/test_stochtree.py 'tests/test_mcmcstep.py::TestMultichain' -k "v1 or not (v2 or v3) or jaxext"
GROUP_bart-v23    := tests/test_BART.py 'tests/test_interface.py::test_equiv_sharding[v4]' -k "v2 or v3 or v4"

GROUPS := misc iface-v5v7 iface-v6 iface-v4 bart-v1 bart-v23

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
TESTS_COMMAND = python -m pytest --cov --cov-context=test --dist=worksteal --durations=1000
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

# pre-commit repos excluded from `update-deps` autoupdate; each pinned rev in
# .pre-commit-config.yaml carries a comment with the reason and unpin condition
PRECOMMIT_PINNED = https://github.com/henryiii/validate-pyproject-schema-store

.PHONY: update-deps
update-deps:
	uv lock --upgrade
	# Update R packages to their latest versions and rewrite renv.lock; snapshot
	# captures the refreshed library (explicit type, from DESCRIPTION). renv's
	# installer reports build failures without raising an R error, so re-check:
	# update(check = TRUE) returns TRUE only when nothing is left to update,
	# and status() that the library and lockfile agree.
	Rscript -e 'renv::update(prompt = FALSE); renv::snapshot(prompt = FALSE); stopifnot(isTRUE(renv::update(check = TRUE)), renv::status()$$synchronized)'
	# --freeze keeps revs pinned to commit SHAs (tags are mutable); autoupdate
	# has no exclude flag, so repos in PRECOMMIT_PINNED are kept back by
	# passing every other remote repo with --repo
	$(UV_RUN) pre-commit autoupdate --freeze $$($(UV_RUN) python -c "import sys, yaml; print(' '.join('--repo ' + r['repo'] for r in yaml.safe_load(open('.pre-commit-config.yaml'))['repos'] if r['repo'] not in ('local', 'meta', *sys.argv[1:])))" $(PRECOMMIT_PINNED))

.PHONY: update-oldest-deps
update-oldest-deps:
	$(UV_RUN) python config/update_python_version.py --bump-date=$(BUMP_PYTHON_VERSION_DATE) --num-supported=$(NUM_SUPPORTED_PYTHON_RELEASES)
	$(UV_RUN) python config/update_oldest_deps.py --min-old-date=$(OLD_DATE) --delay-days=$(OLD_DELAY_DAYS)
	uv lock


################# RELEASE #################

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: check-changelog
check-changelog:
	$(UV_RUN) python config/util.py check_changelog

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
release: check-changelog clean setup update-oldest-deps update-deps check-committed tests tests-single-cpu tests-old docs version-tag build upload gh-release
	@echo "Done!"

.PHONY: version-tag
version-tag: check-committed
	test $(shell git rev-parse --abbrev-ref HEAD) = main
	git fetch --tags
	$(eval VERSION_TAG := v$(shell $(UV_RUN) python config/util.py get_version))
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
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
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
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install bartz $$VERSION from PyPI" && \
	uv tool run --exclude-newer-package="bartz=0 days" --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

# Like `upload`, but the tag stays local: TestPyPI uploads are rehearsals.
.PHONY: upload-test
upload-test: version-tag check-dist smoke-test
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish --check-url=https://test.pypi.org/simple/ --publish-url=https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --exclude-newer-package="bartz=0 days" --index=https://test.pypi.org/simple/ --index-strategy=unsafe-best-match --with="bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: gh-release
gh-release: push-tag
	$(UV_RUN) python config/util.py gh_release


################# BENCHMARKS #################

ASV = $(UV_RUN) python -m asv

.PHONY: asv-machine
asv-machine:
	$(UV_RUN) python config/asv_machine.py

.PHONY: asv-run
asv-run: ASV_REFS = $(shell $(UV_RUN) python config/refs_for_asv.py)
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
