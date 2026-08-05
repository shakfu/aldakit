
define d2-render
d2 $(1) docs/assets/$(basename $(notdir $(1))).$(2)
endef


.PHONY: all sync resync build test clean format lint typecheck check  \
		reset publish publish-test assets fullcheck wheel release \
		coverage test-review golden instruments generated

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
	@uv run python -m pytest tests/ -v

test-review:
	@uv run python -m pytest --review tests/

coverage:
	@uv run python -m pytest --cov-report term-missing:skip-covered --cov=src/aldakit tests/

# Regenerate the committed instrument table from the language docs.
instruments:
	@uv run python scripts/gen_instruments.py

# Regenerate the golden MIDI fixtures. Review the diff: it shows exactly which
# notes changed, so an unintended change to how a score sounds is not silent.
golden:
	@uv run python scripts/gen_golden_midi.py

generated: instruments golden



format:
	@uv run ruff format src/ tests/

lint:
	@uv run ruff check --fix src/ tests/

typecheck:
	@uv run ty check src/aldakit/

check:
	@uv run twine check dist/*

fullcheck: format lint typecheck test

publish-test: check
	@uv run twine upload --verbose --repository testpypi dist/*

publish: check
	@uv run twine upload dist/*

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
