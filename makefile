.PHONY: help test test-test test-example test-lightweight
.PHONY: black black-check flake8
.PHONY: install install-dev install-devtools install-test install-lint install-docs
.PHONY: conda-env
.PHONY: black isort format
.PHONY: black-check isort-check format-check
.PHONY: flake8
.PHONY: pydocstyle-check
.PHONY: darglint-check
.PHONY: build-docs

.DEFAULT: help

help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle on the project"
	@echo "darglint-check"
	@echo "        Run darglint on the project"
	@echo "install"
	@echo "        Install backpack and dependencies"
	@echo "install-dev"
	@echo "        Install all development tools"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "install-docs"
	@echo "        Install only the tools to build/view the docs (included in install-dev)"
	@echo "conda-env"
	@echo "        Create conda environment 'backpack' with dev setup"
	@echo "build-docs"
	@echo "        Build the docs"

###
# Test coverage
test: test-example test-test

test-example:
	@python example/supported.py
	@python example/extend.py
	@python example/extend_with_access_unreduced_loss.py

	@echo "BackPACK runner with SGD on mnist_logreg"
	@python example/run.py mnist_logreg

	@echo "BackPACK runner with SGD on fmnist_2c2d"
	@python example/run.py fmnist_2c2d

	@echo "BackPACK runner with SGD on cifar10_3c3d"
	@python example/run.py cifar10_3c3d --l2_reg 0.0

	@echo "BackPACK runner with SGD on cifar100_allcnnc"
	@python example/run.py cifar100_allcnnc --l2_reg 0.0

test-test:
	@pytest -vx --cov=backobs test

test-lightweight:
	@pytest -vx -k mnist --cov=backobs test
###
# Linter and autoformatter

# Uses black.toml config instead of pyproject.toml to avoid pip issues. See
# - https://github.com/psf/black/issues/683
# - https://github.com/pypa/pip/pull/6370
# - https://pip.pypa.io/en/stable/reference/pip/#pep-517-and-518-support
black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

flake8:
	@flake8 .

pydocstyle-check:
	@pydocstyle --count backobs

darglint-check:
	@darglint --verbosity 2 backobs

isort:
	@isort --apply .

isort-check:
	@isort --check .

format:
	@make black
	@make isort
	@make black-check

format-check: black-check isort-check pydocstyle-check darglint-check

###
# Installation

install:
	@pip install .

install-lint:
	@pip install -r requirements/lint.txt

install-test:
	@pip install -r requirements/test.txt

install-docs:
	@pip install -r requirements/docs.txt

install-devtools:
	@echo "Install dev tools..."
	@pip install -r requirements-dev.txt

install-dev: install-devtools
	@echo "Uninstall existing version of backobs..."
	@pip uninstall backobs
	@echo "Install backobs in editable mode..."
	@pip install -e .

###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml

###
# Documentation
build-docs:
	@cd docs_src/rtd && make html
