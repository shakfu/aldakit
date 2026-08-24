"""Interactive REPL built on the standard-library terminal frontend.

This is the dependency-free counterpart to :mod:`aldakit.repl`. Both drive the
same session state and command handlers from :mod:`aldakit.repl_core`, so the
only thing that differs between them is input and rendering.
"""

from __future__ import annotations

import os
from pathlib import Path

from .constants import DEFAULT_TEMPO, DEFAULT_VIRTUAL_PORT_NAME
from .repl_core import (
    BackendUnavailable,
    ReplBackend,
    build_context,
    close_backend,
    dispatch,
    load_into_session,
    open_backend,
    print_banner,
)

#: Set to a truthy value to use this frontend instead of the prompt_toolkit one.
FRONTEND_ENV_VAR = "ALDAKIT_STDLIB_REPL"


def stdlib_frontend_requested() -> bool:
    """Whether the environment asks for the standard-library frontend."""
    value = os.environ.get(FRONTEND_ENV_VAR, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def run_terminal_repl(
    port_name: str | None = None,
    verbose: bool = False,
    concurrent: bool = True,
    use_audio: bool = False,
    soundfont: str | None = None,
    default_tempo: int = DEFAULT_TEMPO,
    virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME,
    initial_file: str | Path | None = None,
) -> int:
    """Run the interactive alda REPL without prompt_toolkit.

    Takes the same arguments as :func:`aldakit.repl.run_repl` so the two are
    interchangeable at the call site.
    """
    from .terminal.app import run_editor

    try:
        backend, backend_name, supports_concurrent = open_backend(
            port_name=port_name,
            concurrent=concurrent,
            use_audio=use_audio,
            soundfont=soundfont,
            virtual_port_name=virtual_port_name,
        )
    except BackendUnavailable as error:
        print(f"Error: {error}")
        return 1

    print_banner(backend, backend_name, supports_concurrent)

    context = build_context(
        backend,
        supports_concurrent=supports_concurrent,
        verbose=verbose,
        default_tempo=default_tempo,
        virtual_port_name=virtual_port_name,
    )

    # Load a file given on the command line. It is not played: the REPL opens
    # ready to go, and :play starts it.
    if initial_file is not None:
        load_into_session(context, str(initial_file))
        print()

    try:
        run_editor(
            lambda source: dispatch(context, source),
            stop_playback=lambda: _stop(backend),
            poll=_playback_reporter(backend),
            should_exit=lambda: not context.running,
        )
    except KeyboardInterrupt:
        pass
    finally:
        close_backend(backend, supports_concurrent)
    print("Goodbye!")
    return 0


def _stop(backend: ReplBackend) -> None:
    """Stop playback, reporting only when something was actually playing."""
    if backend.is_playing():
        backend.stop()
        print("Stopped")


def _playback_reporter(backend: ReplBackend):
    """Return a poll callback that announces when playback finishes.

    The start of playback needs no announcement -- the user just asked for it
    -- so only the transition back to idle is reported.
    """
    state = {"playing": False}

    def poll() -> str | None:
        playing = backend.is_playing()
        if playing == state["playing"]:
            return None
        state["playing"] = playing
        return None if playing else "(playback finished)"

    return poll
