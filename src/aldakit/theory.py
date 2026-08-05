"""Music theory tables and pure calculations.

Pitch names, scale intervals and key signatures are needed in three places
that have no other reason to know about each other: the MIDI generator (which
applies a key signature to the notes it emits), the compose module (which
builds scales and chords), and the MIDI import path (which names pitches).
They live here so there is one encoding of each fact.

Everything in this module is pure: no AST nodes, no MIDI, no state. Callers
that work with AST nodes extract the strings first and call in.
"""

from __future__ import annotations

# =============================================================================
# Pitch names
# =============================================================================

# Note letter to semitone offset above C.
PITCH_SEMITONES: dict[str, int] = {
    "c": 0,
    "d": 2,
    "e": 4,
    "f": 5,
    "g": 7,
    "a": 9,
    "b": 11,
}

# Semitone offset above C to (letter, accidental), spelling black keys as
# sharps. Index with ``semitone % 12``.
SEMITONE_PITCHES: list[tuple[str, str | None]] = [
    ("c", None),
    ("c", "+"),
    ("d", None),
    ("d", "+"),
    ("e", None),
    ("f", None),
    ("f", "+"),
    ("g", None),
    ("g", "+"),
    ("a", None),
    ("a", "+"),
    ("b", None),
]

# Accidental characters. Alda writes sharps as "+" and flats as "-"; "#" and
# "b" are accepted where a key signature is spelled out in a string.
SHARP_CHARS = ("+", "#")
FLAT_CHARS = ("-", "b")
NATURAL_CHAR = "_"


def accidental_offset(accidentals: str | list[str]) -> int:
    """Total semitone offset of a run of accidentals.

    Naturals contribute nothing: they cancel a key signature rather than
    shifting the pitch.
    """
    offset = 0
    for acc in accidentals:
        if acc in SHARP_CHARS:
            offset += 1
        elif acc in FLAT_CHARS:
            offset -= 1
    return offset


def parse_root(root: str) -> int | None:
    """Semitone (0-11) of a root spelled like ``"c"``, ``"f+"`` or ``"bb"``.

    Returns None when the letter is not a note name.
    """
    if not root:
        return None
    letter = root[0].lower()
    if letter not in PITCH_SEMITONES:
        return None
    semitone = PITCH_SEMITONES[letter]
    if len(root) > 1:
        # "b" reads as a flat only after the letter, never as the note B.
        semitone += accidental_offset(root[1:])
    return semitone % 12


# =============================================================================
# Scales and modes
# =============================================================================

# Intervals from the root, in semitones.
SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    # Major modes
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),  # Same as major
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),  # Natural minor
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    # Minor scales
    "minor": (0, 2, 3, 5, 7, 8, 10),  # Natural minor (aeolian)
    "harmonic-minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic-minor": (0, 2, 3, 5, 7, 9, 11),  # Ascending form
    # Pentatonic scales
    "pentatonic": (0, 2, 4, 7, 9),  # Major pentatonic
    "major-pentatonic": (0, 2, 4, 7, 9),
    "minor-pentatonic": (0, 3, 5, 7, 10),
    # Blues scales
    "blues": (0, 3, 5, 6, 7, 10),
    "major-blues": (0, 2, 3, 4, 7, 9),
    # Other common scales
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "whole-tone": (0, 2, 4, 6, 8, 10),
    "diminished": (0, 2, 3, 5, 6, 8, 9, 11),  # Half-whole
    "diminished-whole-half": (0, 1, 3, 4, 6, 7, 9, 10),  # Whole-half
    "augmented": (0, 3, 4, 7, 8, 11),
    # World scales
    "japanese": (0, 1, 5, 7, 8),  # In scale
    "arabic": (0, 1, 4, 5, 7, 8, 11),  # Double harmonic
    "hungarian-minor": (0, 2, 3, 6, 7, 8, 11),
    "spanish": (0, 1, 4, 5, 7, 8, 10),  # Phrygian dominant
    # Bebop scales
    "bebop-dominant": (0, 2, 4, 5, 7, 9, 10, 11),
    "bebop-major": (0, 2, 4, 5, 7, 8, 9, 11),
}

# Semitone offset of each mode's root within its parent major scale, used to
# find the key signature of a mode on an arbitrary root: D dorian sits two
# semitones above C, so it takes C major's key signature.
MODE_INTERVALS: dict[str, int] = {
    "ionian": 0,  # Same as major
    "dorian": 2,  # 2nd degree of major
    "phrygian": 4,  # 3rd degree
    "lydian": 5,  # 4th degree
    "mixolydian": 7,  # 5th degree
    "aeolian": 9,  # 6th degree (natural minor)
    "locrian": 11,  # 7th degree
}


# =============================================================================
# Key signatures
# =============================================================================

# Key name to {note letter: accidental}, where "+" is sharp and "-" is flat.
KEY_SIGNATURES: dict[str, dict[str, str]] = {
    # Major keys (sharp side)
    "c major": {},
    "g major": {"f": "+"},
    "d major": {"f": "+", "c": "+"},
    "a major": {"f": "+", "c": "+", "g": "+"},
    "e major": {"f": "+", "c": "+", "g": "+", "d": "+"},
    "b major": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+"},
    "f# major": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+"},
    "f+ major": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+"},
    "c# major": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+", "b": "+"},
    "c+ major": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+", "b": "+"},
    # Major keys (flat side)
    "f major": {"b": "-"},
    "bb major": {"b": "-", "e": "-"},
    "b- major": {"b": "-", "e": "-"},
    "eb major": {"b": "-", "e": "-", "a": "-"},
    "e- major": {"b": "-", "e": "-", "a": "-"},
    "ab major": {"b": "-", "e": "-", "a": "-", "d": "-"},
    "a- major": {"b": "-", "e": "-", "a": "-", "d": "-"},
    "db major": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-"},
    "d- major": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-"},
    "gb major": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-"},
    "g- major": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-"},
    "cb major": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-", "f": "-"},
    "c- major": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-", "f": "-"},
    # Minor keys (sharp side) - relative to major
    "a minor": {},
    "e minor": {"f": "+"},
    "b minor": {"f": "+", "c": "+"},
    "f# minor": {"f": "+", "c": "+", "g": "+"},
    "f+ minor": {"f": "+", "c": "+", "g": "+"},
    "c# minor": {"f": "+", "c": "+", "g": "+", "d": "+"},
    "c+ minor": {"f": "+", "c": "+", "g": "+", "d": "+"},
    "g# minor": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+"},
    "g+ minor": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+"},
    "d# minor": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+"},
    "d+ minor": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+"},
    "a# minor": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+", "b": "+"},
    "a+ minor": {"f": "+", "c": "+", "g": "+", "d": "+", "a": "+", "e": "+", "b": "+"},
    # Minor keys (flat side)
    "d minor": {"b": "-"},
    "g minor": {"b": "-", "e": "-"},
    "c minor": {"b": "-", "e": "-", "a": "-"},
    "f minor": {"b": "-", "e": "-", "a": "-", "d": "-"},
    "bb minor": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-"},
    "b- minor": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-"},
    "eb minor": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-"},
    "e- minor": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-"},
    "ab minor": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-", "f": "-"},
    "a- minor": {"b": "-", "e": "-", "a": "-", "d": "-", "g": "-", "c": "-", "f": "-"},
    # Modes on C's white notes; modes on other roots are calculated by
    # mode_key_signature().
    "c ionian": {},
    "d dorian": {},
    "e phrygian": {},
    "f lydian": {},
    "g mixolydian": {},
    "a aeolian": {},
    "b locrian": {},
}

# Parent major key for each semitone, used when calculating modal signatures.
SEMITONE_MAJOR_KEYS: dict[int, str] = {
    0: "c major",
    1: "db major",
    2: "d major",
    3: "eb major",
    4: "e major",
    5: "f major",
    6: "gb major",
    7: "g major",
    8: "ab major",
    9: "a major",
    10: "bb major",
    11: "b major",
}


def key_signature_from_string(spec: str) -> dict[str, str]:
    """Parse a key signature written out as accidentals, e.g. ``"f+ c+ g+"``.

    Unrecognised tokens are ignored, so a partially valid spec yields the
    accidentals it could understand rather than an error.
    """
    key_sig: dict[str, str] = {}
    for token in spec.lower().split():
        if not token:
            continue
        letter = token[0]
        if letter not in PITCH_SEMITONES:
            continue
        accidentals = token[1:]
        if any(c in accidentals for c in SHARP_CHARS):
            key_sig[letter] = "+"
        elif any(c in accidentals for c in FLAT_CHARS):
            key_sig[letter] = "-"
    return key_sig


def key_signature_from_accidental_words(symbols: list[str]) -> dict[str, str]:
    """Parse ``["e", "flat", "b", "flat"]`` into ``{"e": "-", "b": "-"}``."""
    key_sig: dict[str, str] = {}
    i = 0
    while i < len(symbols):
        if symbols[i] in PITCH_SEMITONES and i + 1 < len(symbols):
            letter = symbols[i]
            word = symbols[i + 1]
            if word == "flat":
                key_sig[letter] = "-"
                i += 2
                continue
            if word == "sharp":
                key_sig[letter] = "+"
                i += 2
                continue
        i += 1
    return key_sig


def mode_key_signature(root: str, mode: str) -> dict[str, str] | None:
    """Key signature of a mode on any root, e.g. D dorian -> C major's.

    Returns None when the mode or the root is not recognised.
    """
    if mode not in MODE_INTERVALS:
        return None

    root_semitone = parse_root(root)
    if root_semitone is None:
        return None

    parent_semitone = (root_semitone - MODE_INTERVALS[mode]) % 12
    parent_major = SEMITONE_MAJOR_KEYS.get(parent_semitone)
    if parent_major is None:
        return None
    signature = KEY_SIGNATURES.get(parent_major)
    return None if signature is None else signature.copy()


def key_signature_from_symbols(symbols: list[str]) -> dict[str, str] | None:
    """Resolve a key signature written as symbols in a quoted list.

    Handles ``["g", "minor"]``, ``["c", "ionian"]`` and the spelled-out
    ``["e", "flat", "b", "flat"]`` form. Returns None when nothing matches.
    """
    if len(symbols) < 2:
        return None

    if symbols[1] in ("flat", "sharp"):
        return key_signature_from_accidental_words(symbols)

    named = KEY_SIGNATURES.get(" ".join(symbols))
    if named is not None:
        return named.copy()

    return mode_key_signature(symbols[0], symbols[1])
