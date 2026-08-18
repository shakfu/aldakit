#!/usr/bin/env python3
"""Regenerate the golden audio fixtures for the bundled examples.

The golden MIDI fixtures pin what the generator decided; these pin what it
sounds like. Every example is synthesized with a checksum-pinned SoundFont and
the shape of the resulting audio -- loudness per channel over quarter second
windows, the peak, and the length -- is written to ``tests/golden/audio.json``.

This catches the class of defect that well-formed MIDI hides: an instrument
that never sounds because its program change landed on another part's channel,
a pan that reached the wrong channel, a part that stops early.

Needs the pinned SoundFont, which is not in the repository::

    aldakit soundfont install TimGM6mb
    python scripts/gen_golden_audio.py

Rendering all of the examples takes about a minute. Review the diff: a change
here means scores that used to sound one way now sound another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aldakit import generate_midi, parse  # noqa: E402
from aldakit.midi.render import render_pcm  # noqa: E402
from tests.helpers import (  # noqa: E402
    AUDIO_GAIN,
    AUDIO_SOUNDFONT,
    AUDIO_TAIL_SECONDS,
    EXAMPLES,
    GOLDEN_DIR,
    audio_fingerprint,
    pinned_soundfont,
)


def main() -> int:
    soundfont = pinned_soundfont()
    if soundfont is None:
        print(
            f"The {AUDIO_SOUNDFONT} SoundFont is not installed, or the file "
            f"installed under that name is not the pinned one.\n"
            f"  Run: aldakit soundfont install {AUDIO_SOUNDFONT}",
            file=sys.stderr,
        )
        return 1

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden: dict[str, dict] = {}

    for path in sorted(EXAMPLES.glob("*.alda")):
        sequence = generate_midi(parse(path.read_text(encoding="utf-8"), str(path)))
        pcm, sample_rate, peak = render_pcm(
            sequence, soundfont, gain=AUDIO_GAIN, tail=AUDIO_TAIL_SECONDS
        )
        golden[path.name] = audio_fingerprint(pcm, sample_rate, peak)
        print(f"  {path.name}", file=sys.stderr)

    out = GOLDEN_DIR / "audio.json"
    out.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    seconds = sum(v["frames"] / v["sample_rate"] for v in golden.values())
    print(f"Wrote {out}: {len(golden)} examples, {seconds:.0f}s of audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
