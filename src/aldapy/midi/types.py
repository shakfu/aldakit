"""MIDI types and data structures."""

from dataclasses import dataclass, field
from enum import IntEnum


class GeneralMidiProgram(IntEnum):
    """General MIDI program numbers for common instruments."""

    # Piano
    ACOUSTIC_GRAND_PIANO = 0
    BRIGHT_ACOUSTIC_PIANO = 1
    ELECTRIC_GRAND_PIANO = 2
    HONKY_TONK_PIANO = 3
    ELECTRIC_PIANO_1 = 4
    ELECTRIC_PIANO_2 = 5
    HARPSICHORD = 6
    CLAVINET = 7

    # Chromatic Percussion
    CELESTA = 8
    GLOCKENSPIEL = 9
    MUSIC_BOX = 10
    VIBRAPHONE = 11
    MARIMBA = 12
    XYLOPHONE = 13
    TUBULAR_BELLS = 14
    DULCIMER = 15

    # Organ
    DRAWBAR_ORGAN = 16
    PERCUSSIVE_ORGAN = 17
    ROCK_ORGAN = 18
    CHURCH_ORGAN = 19
    REED_ORGAN = 20
    ACCORDION = 21
    HARMONICA = 22
    TANGO_ACCORDION = 23

    # Guitar
    ACOUSTIC_GUITAR_NYLON = 24
    ACOUSTIC_GUITAR_STEEL = 25
    ELECTRIC_GUITAR_JAZZ = 26
    ELECTRIC_GUITAR_CLEAN = 27
    ELECTRIC_GUITAR_MUTED = 28
    OVERDRIVEN_GUITAR = 29
    DISTORTION_GUITAR = 30
    GUITAR_HARMONICS = 31

    # Bass
    ACOUSTIC_BASS = 32
    ELECTRIC_BASS_FINGER = 33
    ELECTRIC_BASS_PICK = 34
    FRETLESS_BASS = 35
    SLAP_BASS_1 = 36
    SLAP_BASS_2 = 37
    SYNTH_BASS_1 = 38
    SYNTH_BASS_2 = 39

    # Strings
    VIOLIN = 40
    VIOLA = 41
    CELLO = 42
    CONTRABASS = 43
    TREMOLO_STRINGS = 44
    PIZZICATO_STRINGS = 45
    ORCHESTRAL_HARP = 46
    TIMPANI = 47

    # Ensemble
    STRING_ENSEMBLE_1 = 48
    STRING_ENSEMBLE_2 = 49
    SYNTH_STRINGS_1 = 50
    SYNTH_STRINGS_2 = 51
    CHOIR_AAHS = 52
    VOICE_OOHS = 53
    SYNTH_CHOIR = 54
    ORCHESTRA_HIT = 55

    # Brass
    TRUMPET = 56
    TROMBONE = 57
    TUBA = 58
    MUTED_TRUMPET = 59
    FRENCH_HORN = 60
    BRASS_SECTION = 61
    SYNTH_BRASS_1 = 62
    SYNTH_BRASS_2 = 63

    # Reed
    SOPRANO_SAX = 64
    ALTO_SAX = 65
    TENOR_SAX = 66
    BARITONE_SAX = 67
    OBOE = 68
    ENGLISH_HORN = 69
    BASSOON = 70
    CLARINET = 71

    # Pipe
    PICCOLO = 72
    FLUTE = 73
    RECORDER = 74
    PAN_FLUTE = 75
    BLOWN_BOTTLE = 76
    SHAKUHACHI = 77
    WHISTLE = 78
    OCARINA = 79

    # Synth Lead
    LEAD_1_SQUARE = 80
    LEAD_2_SAWTOOTH = 81
    LEAD_3_CALLIOPE = 82
    LEAD_4_CHIFF = 83
    LEAD_5_CHARANG = 84
    LEAD_6_VOICE = 85
    LEAD_7_FIFTHS = 86
    LEAD_8_BASS_LEAD = 87

    # Synth Pad
    PAD_1_NEW_AGE = 88
    PAD_2_WARM = 89
    PAD_3_POLYSYNTH = 90
    PAD_4_CHOIR = 91
    PAD_5_BOWED = 92
    PAD_6_METALLIC = 93
    PAD_7_HALO = 94
    PAD_8_SWEEP = 95

    # Synth Effects
    FX_1_RAIN = 96
    FX_2_SOUNDTRACK = 97
    FX_3_CRYSTAL = 98
    FX_4_ATMOSPHERE = 99
    FX_5_BRIGHTNESS = 100
    FX_6_GOBLINS = 101
    FX_7_ECHOES = 102
    FX_8_SCI_FI = 103

    # Ethnic
    SITAR = 104
    BANJO = 105
    SHAMISEN = 106
    KOTO = 107
    KALIMBA = 108
    BAGPIPE = 109
    FIDDLE = 110
    SHANAI = 111

    # Percussive
    TINKLE_BELL = 112
    AGOGO = 113
    STEEL_DRUMS = 114
    WOODBLOCK = 115
    TAIKO_DRUM = 116
    MELODIC_TOM = 117
    SYNTH_DRUM = 118
    REVERSE_CYMBAL = 119

    # Sound Effects
    GUITAR_FRET_NOISE = 120
    BREATH_NOISE = 121
    SEASHORE = 122
    BIRD_TWEET = 123
    TELEPHONE_RING = 124
    HELICOPTER = 125
    APPLAUSE = 126
    GUNSHOT = 127


# Mapping from Alda instrument names to GM program numbers
INSTRUMENT_PROGRAMS: dict[str, int] = {
    # Piano
    "piano": GeneralMidiProgram.ACOUSTIC_GRAND_PIANO,
    "acoustic-grand-piano": GeneralMidiProgram.ACOUSTIC_GRAND_PIANO,
    "bright-acoustic-piano": GeneralMidiProgram.BRIGHT_ACOUSTIC_PIANO,
    "electric-grand-piano": GeneralMidiProgram.ELECTRIC_GRAND_PIANO,
    "honky-tonk-piano": GeneralMidiProgram.HONKY_TONK_PIANO,
    "electric-piano-1": GeneralMidiProgram.ELECTRIC_PIANO_1,
    "electric-piano-2": GeneralMidiProgram.ELECTRIC_PIANO_2,
    "harpsichord": GeneralMidiProgram.HARPSICHORD,
    "clavinet": GeneralMidiProgram.CLAVINET,
    # Chromatic Percussion
    "celesta": GeneralMidiProgram.CELESTA,
    "glockenspiel": GeneralMidiProgram.GLOCKENSPIEL,
    "music-box": GeneralMidiProgram.MUSIC_BOX,
    "vibraphone": GeneralMidiProgram.VIBRAPHONE,
    "marimba": GeneralMidiProgram.MARIMBA,
    "xylophone": GeneralMidiProgram.XYLOPHONE,
    "tubular-bells": GeneralMidiProgram.TUBULAR_BELLS,
    "dulcimer": GeneralMidiProgram.DULCIMER,
    # Organ
    "organ": GeneralMidiProgram.DRAWBAR_ORGAN,
    "drawbar-organ": GeneralMidiProgram.DRAWBAR_ORGAN,
    "percussive-organ": GeneralMidiProgram.PERCUSSIVE_ORGAN,
    "rock-organ": GeneralMidiProgram.ROCK_ORGAN,
    "church-organ": GeneralMidiProgram.CHURCH_ORGAN,
    "reed-organ": GeneralMidiProgram.REED_ORGAN,
    "accordion": GeneralMidiProgram.ACCORDION,
    "harmonica": GeneralMidiProgram.HARMONICA,
    "tango-accordion": GeneralMidiProgram.TANGO_ACCORDION,
    # Guitar
    "guitar": GeneralMidiProgram.ACOUSTIC_GUITAR_NYLON,
    "acoustic-guitar": GeneralMidiProgram.ACOUSTIC_GUITAR_NYLON,
    "acoustic-guitar-nylon": GeneralMidiProgram.ACOUSTIC_GUITAR_NYLON,
    "acoustic-guitar-steel": GeneralMidiProgram.ACOUSTIC_GUITAR_STEEL,
    "electric-guitar-jazz": GeneralMidiProgram.ELECTRIC_GUITAR_JAZZ,
    "electric-guitar-clean": GeneralMidiProgram.ELECTRIC_GUITAR_CLEAN,
    "electric-guitar-muted": GeneralMidiProgram.ELECTRIC_GUITAR_MUTED,
    "overdriven-guitar": GeneralMidiProgram.OVERDRIVEN_GUITAR,
    "distortion-guitar": GeneralMidiProgram.DISTORTION_GUITAR,
    "electric-guitar-distorted": GeneralMidiProgram.DISTORTION_GUITAR,
    "guitar-harmonics": GeneralMidiProgram.GUITAR_HARMONICS,
    # Bass
    "bass": GeneralMidiProgram.ACOUSTIC_BASS,
    "acoustic-bass": GeneralMidiProgram.ACOUSTIC_BASS,
    "electric-bass": GeneralMidiProgram.ELECTRIC_BASS_FINGER,
    "electric-bass-finger": GeneralMidiProgram.ELECTRIC_BASS_FINGER,
    "electric-bass-pick": GeneralMidiProgram.ELECTRIC_BASS_PICK,
    "fretless-bass": GeneralMidiProgram.FRETLESS_BASS,
    "slap-bass-1": GeneralMidiProgram.SLAP_BASS_1,
    "slap-bass-2": GeneralMidiProgram.SLAP_BASS_2,
    "synth-bass-1": GeneralMidiProgram.SYNTH_BASS_1,
    "synth-bass-2": GeneralMidiProgram.SYNTH_BASS_2,
    # Strings
    "violin": GeneralMidiProgram.VIOLIN,
    "viola": GeneralMidiProgram.VIOLA,
    "cello": GeneralMidiProgram.CELLO,
    "contrabass": GeneralMidiProgram.CONTRABASS,
    "double-bass": GeneralMidiProgram.CONTRABASS,
    "tremolo-strings": GeneralMidiProgram.TREMOLO_STRINGS,
    "pizzicato-strings": GeneralMidiProgram.PIZZICATO_STRINGS,
    "harp": GeneralMidiProgram.ORCHESTRAL_HARP,
    "orchestral-harp": GeneralMidiProgram.ORCHESTRAL_HARP,
    "timpani": GeneralMidiProgram.TIMPANI,
    # Ensemble
    "string-ensemble-1": GeneralMidiProgram.STRING_ENSEMBLE_1,
    "string-ensemble-2": GeneralMidiProgram.STRING_ENSEMBLE_2,
    "synth-strings-1": GeneralMidiProgram.SYNTH_STRINGS_1,
    "synth-strings-2": GeneralMidiProgram.SYNTH_STRINGS_2,
    "choir": GeneralMidiProgram.CHOIR_AAHS,
    "choir-aahs": GeneralMidiProgram.CHOIR_AAHS,
    "voice-oohs": GeneralMidiProgram.VOICE_OOHS,
    "synth-choir": GeneralMidiProgram.SYNTH_CHOIR,
    "orchestra-hit": GeneralMidiProgram.ORCHESTRA_HIT,
    # Brass
    "trumpet": GeneralMidiProgram.TRUMPET,
    "trombone": GeneralMidiProgram.TROMBONE,
    "tuba": GeneralMidiProgram.TUBA,
    "muted-trumpet": GeneralMidiProgram.MUTED_TRUMPET,
    "french-horn": GeneralMidiProgram.FRENCH_HORN,
    "brass-section": GeneralMidiProgram.BRASS_SECTION,
    "synth-brass-1": GeneralMidiProgram.SYNTH_BRASS_1,
    "synth-brass-2": GeneralMidiProgram.SYNTH_BRASS_2,
    # Reed
    "soprano-sax": GeneralMidiProgram.SOPRANO_SAX,
    "alto-sax": GeneralMidiProgram.ALTO_SAX,
    "tenor-sax": GeneralMidiProgram.TENOR_SAX,
    "baritone-sax": GeneralMidiProgram.BARITONE_SAX,
    "oboe": GeneralMidiProgram.OBOE,
    "english-horn": GeneralMidiProgram.ENGLISH_HORN,
    "bassoon": GeneralMidiProgram.BASSOON,
    "clarinet": GeneralMidiProgram.CLARINET,
    # Pipe
    "piccolo": GeneralMidiProgram.PICCOLO,
    "flute": GeneralMidiProgram.FLUTE,
    "recorder": GeneralMidiProgram.RECORDER,
    "pan-flute": GeneralMidiProgram.PAN_FLUTE,
    "blown-bottle": GeneralMidiProgram.BLOWN_BOTTLE,
    "shakuhachi": GeneralMidiProgram.SHAKUHACHI,
    "whistle": GeneralMidiProgram.WHISTLE,
    "ocarina": GeneralMidiProgram.OCARINA,
    # Synth
    "midi-synth-pad-new-age": GeneralMidiProgram.PAD_1_NEW_AGE,
}


@dataclass
class MidiNote:
    """A MIDI note event."""

    pitch: int  # MIDI note number (0-127)
    velocity: int  # Note velocity (0-127)
    start_time: float  # Start time in seconds
    duration: float  # Duration in seconds
    channel: int = 0  # MIDI channel (0-15)


@dataclass
class MidiProgramChange:
    """A MIDI program change event."""

    program: int  # Program number (0-127)
    time: float  # Time in seconds
    channel: int = 0  # MIDI channel (0-15)


@dataclass
class MidiControlChange:
    """A MIDI control change event."""

    control: int  # Control number (0-127)
    value: int  # Control value (0-127)
    time: float  # Time in seconds
    channel: int = 0  # MIDI channel (0-15)


@dataclass
class MidiTempoChange:
    """A tempo change event."""

    bpm: float  # Beats per minute
    time: float  # Time in seconds


@dataclass
class MidiSequence:
    """A complete MIDI sequence."""

    notes: list[MidiNote] = field(default_factory=list)
    program_changes: list[MidiProgramChange] = field(default_factory=list)
    control_changes: list[MidiControlChange] = field(default_factory=list)
    tempo_changes: list[MidiTempoChange] = field(default_factory=list)
    ticks_per_beat: int = 480

    def duration(self) -> float:
        """Return the total duration in seconds."""
        if not self.notes:
            return 0.0
        return max(n.start_time + n.duration for n in self.notes)


# Note letter to semitone offset (relative to C)
NOTE_OFFSETS: dict[str, int] = {
    "c": 0,
    "d": 2,
    "e": 4,
    "f": 5,
    "g": 7,
    "a": 9,
    "b": 11,
}


def note_to_midi(letter: str, octave: int, accidentals: list[str]) -> int:
    """Convert a note letter, octave, and accidentals to MIDI note number.

    C4 = MIDI 60 (middle C).
    """
    base = NOTE_OFFSETS[letter.lower()]
    midi_note = 12 * (octave + 1) + base

    for acc in accidentals:
        if acc == "+":  # sharp
            midi_note += 1
        elif acc == "-":  # flat
            midi_note -= 1
        # "_" (natural) has no effect in this context

    return max(0, min(127, midi_note))
