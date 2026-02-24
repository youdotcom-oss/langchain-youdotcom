.PHONY: all format lint type test tests integration_tests check_imports help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

PYTHON_FILES = langchain_youdotcom tests scripts
MYPY_CACHE = .mypy_cache

TEST_FILE ?= tests/unit_tests/
integration_tests: TEST_FILE=tests/integration_tests/

test tests integration_tests:
	uv run --all-groups pytest $(TEST_FILE)

check_imports: $(shell find langchain_youdotcom -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_youdotcom
lint_tests: PYTHON_FILES=tests

help:
	@echo "Available targets:"
	@echo "  format            - Format code with ruff"
	@echo "  lint              - Run ruff check, ruff format --diff, and mypy"
	@echo "  type              - Run mypy type checking"
	@echo "  test              - Run unit tests"
	@echo "  integration_tests - Run integration tests"
	@echo "  check_imports     - Verify all modules are importable"
