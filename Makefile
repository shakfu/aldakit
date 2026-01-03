
define d2-render
d2 $(1) docs/assets/$(basename $(notdir $(1))).$(2)
endef


.PHONY: all sync resync build test clean format lint typecheck check  \
		reset publish publish-test assets fullcheck wheel release

all: sync

sync:
	@uv sync --reinstall-package aldapy

resync: reset sync

build:
	@uv build

wheel:
	@uv build --wheel

release:
	@uv build --sdist
	@uv build --wheel --python 3.10
	@uv build --wheel --python 3.11
	@uv build --wheel --python 3.12
	@uv build --wheel --python 3.13
	@uv build --wheel --python 3.14

test:
	@uv run pytest tests/ -v

format:
	@uv run ruff format src/ tests/

lint:
	@uv run ruff check --fix src/ tests/

typecheck:
	@uv run ty check src/aldapy/

check:
	@uv run twine check dist/*

fullcheck: format lint typecheck test

publish-test: build check
	@uv run twine upload --repository testpypi dist/*

publish: build check
	@uv run twine upload dist/*

assets:
	@mkdir -p docs/assets
	@$(foreach f,$(wildcard docs/*.d2),$(call d2-render,$(f),svg);)
	@$(foreach f,$(wildcard docs/*.d2),$(call d2-render,$(f),png);)
	@$(foreach f,$(wildcard docs/*.d2),$(call d2-render,$(f),pdf);)

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

reset: clean
	@rm -rf build dist .venv
	@rm -rf .pytest_cache .ruff_cache
