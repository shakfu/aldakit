
define d2-render
d2 $(1) docs/assets/$(basename $(notdir $(1))).$(2)
endef


.PHONY: all sync resync build test clean format lint fix typecheck check  \
		reset publish publish-test assets qa wheel release \
		coverage docs docs-serve docs-deploy \
		golden golden-audio soundfont test-audio instruments generated

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

coverage:
	@uv run python -m pytest --cov-report term-missing:skip-covered --cov=src/aldakit tests/

# Regenerate the committed instrument table from the language docs.
instruments:
	@uv run python scripts/gen_instruments.py

# Regenerate the golden MIDI fixtures. Review the diff: it shows exactly which
# notes changed, so an unintended change to how a score sounds is not silent.
golden:
	@uv run python scripts/gen_golden_midi.py

# Regenerate the golden audio fixtures. Renders every example with the pinned
# SoundFont, which takes about a minute; run 'make soundfont' first if it is
# not installed. Not part of 'generated', which must work without a download.
golden-audio:
	@uv run python scripts/gen_golden_audio.py

# Download the SoundFont the audio fixtures are pinned to.
soundfont:
	@uv run aldakit soundfont install TimGM6mb

# Run the tests that need audio, failing rather than skipping if the pinned
# SoundFont is missing. This is what CI runs.
test-audio:
	@ALDAKIT_REQUIRE_AUDIO_FIXTURES=1 uv run python -m pytest \
		tests/test_golden_audio.py tests/test_render.py -q

generated: instruments golden



format:
	@uv run ruff format src/ tests/

# Read-only, and the same command CI runs. 'make fix' is the one that edits.
lint:
	@uv run ruff check src/ tests/ scripts/

fix:
	@uv run ruff check --fix src/ tests/ scripts/

typecheck:
	@uv run ty check src/aldakit/

# What CI checks, in the order CI checks it. A target named for quality
# assurance that never ran the tests was the wrong shape.
qa: lint typecheck test

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
