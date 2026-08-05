"""Centralized constants and default values for aldakit."""

# =============================================================================
# DEFAULT VALUES
# =============================================================================

# MIDI Virtual Port
DEFAULT_VIRTUAL_PORT_NAME = "AldakitMIDI"

# Music defaults
DEFAULT_TEMPO = 120  # BPM
DEFAULT_OCTAVE = 4
DEFAULT_VOLUME = 69  # MIDI velocity (0-127), ~54% of max, corresponds to mf
DEFAULT_QUANTIZATION = 0.9  # Note length as fraction of duration (0.0-1.0)
DEFAULT_DURATION = 1.0  # Duration in beats

# Backend
DEFAULT_BACKEND = "midi"

# =============================================================================
# MIDI PROTOCOL CONSTANTS
# =============================================================================

# Channel limits
MIDI_MAX_CHANNELS = 16  # Channels 0-15
MIDI_DRUM_CHANNEL = 9  # Channel 10 (0-indexed) is reserved for drums

# Value limits
MIDI_MAX_VELOCITY = 127
MIDI_MIN_VELOCITY = 0
MIDI_MAX_CONTROL_VALUE = 127
MIDI_MAX_NOTE = 127
MIDI_MIN_NOTE = 0

# Status bytes (upper nibble)
MIDI_STATUS_NOTE_OFF = 0x80
MIDI_STATUS_NOTE_ON = 0x90
MIDI_STATUS_CONTROL_CHANGE = 0xB0
MIDI_STATUS_PROGRAM_CHANGE = 0xC0

# Bit masks
MIDI_CHANNEL_MASK = 0x0F  # Lower 4 bits for channel
MIDI_DATA_MASK = 0x7F  # Lower 7 bits for data bytes

# Control Change numbers
MIDI_CC_VOLUME = 7  # Channel volume, what Alda calls track-volume
MIDI_CC_PAN = 10
MIDI_CC_ALL_NOTES_OFF = 123

# =============================================================================
# PLAYBACK & CONCURRENCY
# =============================================================================

# Concurrent playback
MAX_PLAYBACK_SLOTS = 8

# Timing intervals (seconds)
POLL_INTERVAL_DEFAULT = 0.05
POLL_INTERVAL_PLAYBACK = 0.1
THREAD_JOIN_TIMEOUT = 0.5
PLAYBACK_SLEEP_THRESHOLD = 0.01
SEQUENTIAL_MODE_SLEEP = 0.01

# =============================================================================
# TRANSCRIPTION DEFAULTS
# =============================================================================

DEFAULT_RECORDING_DURATION = 10.0  # seconds
DEFAULT_QUANTIZE_GRID = 0.25  # 16th notes
DEFAULT_SWING_RATIO = 2.0 / 3.0  # ~0.666
SWING_RATIO_MIN = 0.0  # exclusive
SWING_RATIO_MAX = 1.0  # exclusive

# =============================================================================
# REPL & UI
# =============================================================================

REPL_PROMPT = "aldakit> "
REPL_CONTINUATION_PROMPT = "  ... "
REPL_HISTORY_FILENAME = ".alda_history"
REPL_COMPLETION_MIN_WORD_LENGTH = 3
REPL_INSTRUMENT_COLUMNS = 4

# =============================================================================
# TEMPO & DURATION CALCULATIONS
# =============================================================================

SECONDS_PER_MINUTE = 60.0
MILLISECONDS_PER_SECOND = 1000.0
BEATS_PER_WHOLE_NOTE = 4.0

# =============================================================================
# DYNAMICS VELOCITY MAPPING
# =============================================================================

# Maps dynamic markings to MIDI velocity values (0-127).
#
# Alda defines dynamics as volume levels on a 0-100 scale, and volume maps to
# velocity as velocity = volume * 127 / 100. The comment on each entry is the
# volume it comes from.
DYNAMICS_VELOCITY: dict[str, int] = {
    "pppppp": 1,  # vol=1
    "ppppp": 10,  # vol=8
    "pppp": 20,  # vol=16
    "ppp": 30,  # vol=24
    "pp": 39,  # vol=31
    "p": 50,  # vol=39
    "mp": 58,  # vol=46
    "mf": 69,  # vol=54, the default
    "f": 79,  # vol=62
    "ff": 88,  # vol=69
    "fff": 98,  # vol=77
    "ffff": 108,  # vol=85
    "fffff": 117,  # vol=92
    "ffffff": 127,  # vol=100
}

# =============================================================================
# SOUNDFONT DISCOVERY
# =============================================================================

SOUNDFONT_ENV_VAR = "ALDAKIT_SOUNDFONT"
DEFAULT_SOUNDFONT_GAIN = 1.0

# The filenames searched for on disk live with the discovery code, in
# aldakit.midi.soundfont. Accidental characters, scale intervals, mode
# intervals and key signatures live in aldakit.theory.
