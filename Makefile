
define d2-render
d2 $(1) docs/assets/$(basename $(notdir $(1))).$(2)
endef


.PHONY: all sync resync build test clean format lint typecheck check  \
		reset publish publish-test assets qa wheel release \
		coverage test-review docs docs-serve docs-deploy

all: sync

sync:
	@uv sync --reinstall-package aldakit

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

test-review:
	@uv run pytest --review tests/

coverage:
	@uv run pytest --cov-report term-missing:skip-covered --cov=src/aldakit tests/



format:
	@uv run ruff format src/ tests/

lint:
	@uv run ruff check --fix src/ tests/

typecheck:
	@uv run ty check src/aldakit/

qa: lint typecheck format

check:
	@uv run twine check dist/*

publish-test: check
	@uv run twine upload --verbose --repository testpypi dist/*

publish: check
	@uv run twine upload dist/*

docs:
	@uv run mkdocs build

docs-serve:
	@uv run mkdocs serve

docs-deploy:
	@uv run mkdocs gh-deploy --force

assets:
	@mkdir -p docs/assets
	@$(foreach f,$(wildcard docs/*.d2),$(call d2-render,$(f),svg);)
	@$(foreach f,$(wildcard docs/*.d2),$(call d2-render,$(f),pdf);)
	@$(foreach f,$(wildcard docs/*.d2),$(call d2-render,$(f),png);)

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

reset: clean
	@rm -rf build dist .venv
	@rm -rf .pytest_cache .ruff_cache
