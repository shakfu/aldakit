"""MIDI generator that converts an Alda AST to MIDI events."""

from dataclasses import dataclass, field

from ..ast_nodes import (
    ASTNode,
    AtMarkerNode,
    BarlineNode,
    BracketedSequenceNode,
    ChordNode,
    CramNode,
    DurationNode,
    EventSequenceNode,
    LispListNode,
    LispNumberNode,
    LispQuotedNode,
    LispStringNode,
    LispSymbolNode,
    MarkerNode,
    NoteLengthMsNode,
    NoteLengthNode,
    NoteLengthSecondsNode,
    NoteNode,
    OctaveDownNode,
    OctaveSetNode,
    OctaveUpNode,
    OnRepetitionsNode,
    PartNode,
    RepeatNode,
    RestNode,
    RootNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    VoiceGroupNode,
)
from ..midi.types import (
    INSTRUMENT_PROGRAMS,
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    MidiTempoChange,
    note_to_midi,
)


# Key signature definitions: maps key name to dict of {note: accidental}
# Accidentals: "+" for sharp, "-" for flat
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
    # Modes (based on C major)
    "c ionian": {},
    "d dorian": {},
    "e phrygian": {},
    "f lydian": {},
    "g mixolydian": {},
    "a aeolian": {},
    "b locrian": {},
    # Modes on other roots would need calculation, but these are the common ones
    # For modes, the key signature is determined by the parent major scale
}

# Mode intervals relative to major (for calculating modes on any root)
MODE_INTERVALS: dict[str, int] = {
    "ionian": 0,  # Same as major
    "dorian": 2,  # 2nd degree of major
    "phrygian": 4,  # 3rd degree
    "lydian": 5,  # 4th degree
    "mixolydian": 7,  # 5th degree
    "aeolian": 9,  # 6th degree (natural minor)
    "locrian": 11,  # 7th degree
}


@dataclass
class PartState:
    """State for a single part/instrument."""

    octave: int = 4
    tempo: float = 120.0  # BPM
    volume: int = 69  # 0-127, default mf (54% of 127)
    quantization: float = 0.9  # 0.0-1.0, affects note duration
    default_duration: float = 1.0  # Duration in beats (quarter note = 1 beat)
    current_time: float = 0.0  # Current time in seconds
    channel: int = 0
    program: int = 0
    key_signature: dict[str, str] = field(default_factory=dict)  # note -> accidental
    transpose: int = 0  # Transposition in semitones


@dataclass
class GeneratorState:
    """Global state for the MIDI generator."""

    global_tempo: float = 120.0
    variables: dict[str, EventSequenceNode] = field(default_factory=dict)
    markers: dict[str, float] = field(default_factory=dict)  # marker -> time in seconds
    parts: dict[str, PartState] = field(default_factory=dict)
    current_parts: list[str] = field(default_factory=list)  # Active parts (multi-instrument support)
    next_channel: int = 0
    repetition_number: int = 1  # Current repetition when in a repeat loop


class MidiGenerator:
    """Generates MIDI events from an Alda AST."""

    def __init__(self) -> None:
        self.sequence = MidiSequence()
        self.state = GeneratorState()

    def generate(self, ast: RootNode) -> MidiSequence:
        """Generate a MIDI sequence from an Alda AST.

        Args:
            ast: The root node of the Alda AST.

        Returns:
            A MidiSequence containing all MIDI events.
        """
        self.sequence = MidiSequence()
        self.state = GeneratorState()

        # Add initial tempo
        self.sequence.tempo_changes.append(
            MidiTempoChange(bpm=self.state.global_tempo, time=0.0)
        )

        # Process all children
        for child in ast.children:
            self._process_node(child)

        # Sort events by time
        self.sequence.notes.sort(key=lambda n: n.start_time)
        self.sequence.program_changes.sort(key=lambda p: p.time)
        self.sequence.tempo_changes.sort(key=lambda t: t.time)

        return self.sequence

    def _get_part_state(self) -> PartState:
        """Get the current part state (first active part), creating default if needed."""
        if not self.state.current_parts:
            # Create implicit part
            self.state.current_parts = ["_default"]
            self.state.parts["_default"] = PartState(
                channel=self.state.next_channel,
                program=0,
            )
            self.state.next_channel = min(15, self.state.next_channel + 1)

        return self.state.parts[self.state.current_parts[0]]

    def _get_all_part_states(self) -> list[PartState]:
        """Get all currently active part states."""
        if not self.state.current_parts:
            return [self._get_part_state()]
        return [self.state.parts[name] for name in self.state.current_parts]

    def _process_node(self, node: ASTNode) -> None:
        """Process an AST node."""
        if isinstance(node, PartNode):
            self._process_part(node)
        elif isinstance(node, EventSequenceNode):
            self._process_event_sequence(node)
        elif isinstance(node, NoteNode):
            self._process_note(node)
        elif isinstance(node, RestNode):
            self._process_rest(node)
        elif isinstance(node, ChordNode):
            self._process_chord(node)
        elif isinstance(node, OctaveSetNode):
            for part in self._get_all_part_states():
                part.octave = node.octave
        elif isinstance(node, OctaveUpNode):
            for part in self._get_all_part_states():
                part.octave += 1
        elif isinstance(node, OctaveDownNode):
            for part in self._get_all_part_states():
                part.octave -= 1
        elif isinstance(node, BarlineNode):
            pass  # Barlines are purely visual
        elif isinstance(node, LispListNode):
            self._process_lisp_list(node)
        elif isinstance(node, VariableDefinitionNode):
            self._process_variable_definition(node)
        elif isinstance(node, VariableReferenceNode):
            self._process_variable_reference(node)
        elif isinstance(node, MarkerNode):
            self._process_marker(node)
        elif isinstance(node, AtMarkerNode):
            self._process_at_marker(node)
        elif isinstance(node, VoiceGroupNode):
            self._process_voice_group(node)
        elif isinstance(node, CramNode):
            self._process_cram(node)
        elif isinstance(node, RepeatNode):
            self._process_repeat(node)
        elif isinstance(node, OnRepetitionsNode):
            self._process_on_repetitions(node)
        elif isinstance(node, BracketedSequenceNode):
            self._process_event_sequence(node.events)

    def _process_part(self, node: PartNode) -> None:
        """Process a part declaration and its events."""
        # Get instrument name(s)
        names = node.declaration.names
        alias = node.declaration.alias

        # For multi-instrument parts (violin/viola/cello), create a part for each
        # The alias applies to the group but each instrument gets its own channel
        active_parts = []

        for i, name in enumerate(names):
            # Use alias+index for group naming, or just instrument name
            if alias and len(names) > 1:
                part_name = f"{alias}_{i}"
            elif alias:
                part_name = alias
            else:
                part_name = name

            # Create or get part state
            if part_name not in self.state.parts:
                # Determine MIDI program from instrument name
                normalized = name.lower().replace("_", "-")
                program = INSTRUMENT_PROGRAMS.get(normalized, 0)

                channel = self.state.next_channel
                self.state.next_channel = min(15, self.state.next_channel + 1)

                self.state.parts[part_name] = PartState(
                    channel=channel,
                    program=program,
                    tempo=self.state.global_tempo,
                )

                # Add program change
                self.sequence.program_changes.append(
                    MidiProgramChange(
                        program=program,
                        time=0.0,
                        channel=channel,
                    )
                )

            active_parts.append(part_name)

        self.state.current_parts = active_parts

        # Process events (will be applied to all active parts)
        self._process_event_sequence(node.events)

    def _process_event_sequence(self, node: EventSequenceNode) -> None:
        """Process a sequence of events."""
        for event in node.events:
            self._process_node(event)

    def _process_note(self, node: NoteNode, is_chord: bool = False) -> float:
        """Process a note, returning its duration in seconds.

        Args:
            node: The note node.
            is_chord: If True, don't advance time after the note.

        Returns:
            Duration of the note in seconds.
        """
        duration_secs = 0.0

        # Process note for each active part (multi-instrument support)
        for part in self._get_all_part_states():
            # Determine accidentals: use explicit accidentals, or key signature, or none
            accidentals = node.accidentals
            if not accidentals:
                # No explicit accidentals - check key signature
                letter = node.letter.lower()
                if letter in part.key_signature:
                    accidentals = [part.key_signature[letter]]
            elif "_" in accidentals:
                # Natural sign explicitly cancels key signature
                accidentals = []

            # Calculate MIDI note number
            midi_note = note_to_midi(node.letter, part.octave, accidentals)

            # Apply transposition
            if part.transpose != 0:
                midi_note = max(0, min(127, midi_note + part.transpose))

            # Calculate duration
            duration_beats = self._calculate_duration(node.duration, part)
            duration_secs = self._beats_to_seconds(duration_beats, part.tempo)

            # Apply quantization (affects actual note length, not timing)
            if node.slurred:
                actual_duration = duration_secs  # Full duration for slurred notes
            else:
                actual_duration = duration_secs * part.quantization

            # Create MIDI note
            midi_note_event = MidiNote(
                pitch=midi_note,
                velocity=part.volume,
                start_time=part.current_time,
                duration=actual_duration,
                channel=part.channel,
            )
            self.sequence.notes.append(midi_note_event)

            # Update default duration if specified
            if node.duration is not None:
                part.default_duration = duration_beats

            # Advance time (unless in chord)
            if not is_chord:
                part.current_time += duration_secs

        return duration_secs

    def _process_rest(self, node: RestNode) -> None:
        """Process a rest."""
        # Process rest for each active part (multi-instrument support)
        for part in self._get_all_part_states():
            duration_beats = self._calculate_duration(node.duration, part)
            duration_secs = self._beats_to_seconds(duration_beats, part.tempo)

            # Update default duration if specified
            if node.duration is not None:
                part.default_duration = duration_beats

            # Advance time
            part.current_time += duration_secs

    def _process_chord(self, node: ChordNode) -> None:
        """Process a chord (simultaneous notes)."""
        # Save start times for all active parts
        all_parts = self._get_all_part_states()
        start_times = {id(p): p.current_time for p in all_parts}
        max_duration = 0.0

        for item in node.notes:
            if isinstance(item, NoteNode):
                duration = self._process_note(item, is_chord=True)
                max_duration = max(max_duration, duration)
            elif isinstance(item, OctaveSetNode):
                for part in all_parts:
                    part.octave = item.octave
            elif isinstance(item, OctaveUpNode):
                for part in all_parts:
                    part.octave += 1
            elif isinstance(item, OctaveDownNode):
                for part in all_parts:
                    part.octave -= 1
            elif isinstance(item, LispListNode):
                self._process_lisp_list(item)

        # Advance time by the longest note for all parts
        for part in all_parts:
            part.current_time = start_times[id(part)] + max_duration

    def _process_lisp_list(self, node: LispListNode) -> None:
        """Process a Lisp S-expression (attribute setting)."""
        if not node.elements:
            return

        # Get the function name
        first = node.elements[0]
        if not isinstance(first, LispSymbolNode):
            return

        func_name = first.name.lower()
        args = node.elements[1:]

        # Get all active parts for multi-instrument support
        all_parts = self._get_all_part_states()

        if func_name in ("tempo", "tempo!"):
            # Set tempo
            if args and isinstance(args[0], LispNumberNode):
                new_tempo = float(args[0].value)
                if func_name == "tempo!":
                    # Global tempo
                    self.state.global_tempo = new_tempo
                    for p in self.state.parts.values():
                        p.tempo = new_tempo
                else:
                    for part in all_parts:
                        part.tempo = new_tempo
                self.sequence.tempo_changes.append(
                    MidiTempoChange(bpm=new_tempo, time=all_parts[0].current_time)
                )

        elif func_name in ("vol", "volume", "vol!", "volume!"):
            # Set volume (0-100 -> 0-127)
            if args and isinstance(args[0], LispNumberNode):
                vol = int(args[0].value)
                velocity = min(127, max(0, int(vol * 127 / 100)))
                for part in all_parts:
                    part.volume = velocity

        elif func_name in ("quant", "quantize", "quantization"):
            # Set quantization (0-100 -> 0.0-1.0)
            if args and isinstance(args[0], LispNumberNode):
                quant = float(args[0].value)
                quantization = max(0.0, min(1.0, quant / 100.0))
                for part in all_parts:
                    part.quantization = quantization

        elif func_name == "panning":
            # Set panning (0-100 -> 0-127)
            if args and isinstance(args[0], LispNumberNode):
                pan = int(args[0].value)
                pan_value = min(127, max(0, int(pan * 127 / 100)))
                from .types import MidiControlChange

                for part in all_parts:
                    self.sequence.control_changes.append(
                        MidiControlChange(
                            control=10,  # Pan control
                            value=pan_value,
                            time=part.current_time,
                            channel=part.channel,
                        )
                    )

        elif func_name in ("octave", "octave!"):
            # Set octave - can be number or quoted symbol ('up, 'down)
            if args:
                if isinstance(args[0], LispNumberNode):
                    octave = int(args[0].value)
                    for part in all_parts:
                        part.octave = octave
                elif isinstance(args[0], LispQuotedNode):
                    # Handle 'up and 'down
                    if isinstance(args[0].value, LispSymbolNode):
                        symbol = args[0].value.name.lower()
                        if symbol == "up":
                            for part in all_parts:
                                part.octave += 1
                        elif symbol == "down":
                            for part in all_parts:
                                part.octave -= 1
                elif isinstance(args[0], LispSymbolNode):
                    # Handle unquoted up/down (non-standard but convenient)
                    symbol = args[0].name.lower()
                    if symbol == "up":
                        for part in all_parts:
                            part.octave += 1
                    elif symbol == "down":
                        for part in all_parts:
                            part.octave -= 1

        # Dynamic markings
        elif func_name in (
            "pppppp",
            "ppppp",
            "pppp",
            "ppp",
            "pp",
            "p",
            "mp",
            "mf",
            "f",
            "ff",
            "fff",
            "ffff",
            "fffff",
            "ffffff",
        ):
            # Official Alda dynamics: volume 0-100 maps to velocity 0-127
            # velocity = volume * 127 / 100
            dynamics = {
                "pppppp": 1,    # vol=1
                "ppppp": 10,   # vol=8
                "pppp": 20,    # vol=16
                "ppp": 30,     # vol=24
                "pp": 39,      # vol=31
                "p": 50,       # vol=39
                "mp": 58,      # vol=46
                "mf": 69,      # vol=54
                "f": 79,       # vol=62
                "ff": 88,      # vol=69
                "fff": 98,     # vol=77
                "ffff": 108,   # vol=85
                "fffff": 117,  # vol=92
                "ffffff": 127, # vol=100
            }
            velocity = dynamics.get(func_name, 69)
            for part in all_parts:
                part.volume = velocity

        elif func_name in ("key-sig", "key-signature", "key-sig!", "key-signature!"):
            # Set key signature
            key_sig = self._parse_key_signature(args)
            if key_sig is not None:
                if func_name.endswith("!"):
                    # Global key signature
                    for p in self.state.parts.values():
                        p.key_signature = key_sig.copy()
                else:
                    for part in all_parts:
                        part.key_signature = key_sig.copy()

        elif func_name in ("transpose", "transpose!"):
            # Set transposition in semitones
            if args and isinstance(args[0], LispNumberNode):
                semitones = int(args[0].value)
                if func_name.endswith("!"):
                    # Global transpose
                    for p in self.state.parts.values():
                        p.transpose = semitones
                else:
                    for part in all_parts:
                        part.transpose = semitones

    def _parse_key_signature(self, args: list) -> dict[str, str] | None:
        """Parse key signature from S-expression arguments.

        Supports formats:
        - String: "f+ c+ g+" (explicit accidentals)
        - Quoted list: '(g minor), '(c ionian), '(e (flat) b (flat))
        """
        if not args:
            return None

        arg = args[0]

        # String format: "f+ c+ g+"
        if isinstance(arg, LispStringNode):
            return self._parse_key_sig_string(arg.value)

        # Quoted list format: '(g minor)
        if isinstance(arg, LispQuotedNode):
            return self._parse_key_sig_quoted(arg.value)

        return None

    def _parse_key_sig_string(self, s: str) -> dict[str, str]:
        """Parse key signature from string format like 'f+ c+ g+'."""
        key_sig: dict[str, str] = {}
        tokens = s.lower().split()

        for token in tokens:
            if not token:
                continue
            note = token[0]
            if note in "abcdefg":
                accidentals = token[1:]
                if "+" in accidentals or "#" in accidentals:
                    key_sig[note] = "+"
                elif "-" in accidentals or "b" in accidentals:
                    key_sig[note] = "-"

        return key_sig

    def _parse_key_sig_quoted(self, node: LispListNode) -> dict[str, str] | None:
        """Parse key signature from quoted list format.

        Formats:
        - (g minor) - key name
        - (c ionian) - mode
        - (e (flat) b (flat)) - explicit accidentals
        """
        if not node.elements:
            return None

        # Extract symbols from the list
        symbols = []
        i = 0
        while i < len(node.elements):
            elem = node.elements[i]
            if isinstance(elem, LispSymbolNode):
                symbols.append(elem.name.lower())
            elif isinstance(elem, LispListNode):
                # Nested list like (flat) or (sharp)
                if elem.elements and isinstance(elem.elements[0], LispSymbolNode):
                    symbols.append(elem.elements[0].name.lower())
            i += 1

        if not symbols:
            return None

        # Check for explicit accidentals format: e flat b flat
        if len(symbols) >= 2 and symbols[1] in ("flat", "sharp"):
            return self._parse_explicit_accidentals(symbols)

        # Check for key name: g minor, d major, c ionian
        if len(symbols) >= 2:
            key_name = " ".join(symbols)
            if key_name in KEY_SIGNATURES:
                return KEY_SIGNATURES[key_name].copy()

            # Try with root + mode/quality
            root = symbols[0]
            quality = symbols[1]

            # Handle modes on any root
            if quality in MODE_INTERVALS:
                return self._calculate_mode_key_sig(root, quality)

        return None

    def _parse_explicit_accidentals(self, symbols: list[str]) -> dict[str, str]:
        """Parse explicit accidentals like: e flat b flat."""
        key_sig: dict[str, str] = {}
        i = 0
        while i < len(symbols):
            if symbols[i] in "abcdefg" and i + 1 < len(symbols):
                note = symbols[i]
                acc = symbols[i + 1]
                if acc == "flat":
                    key_sig[note] = "-"
                    i += 2
                elif acc == "sharp":
                    key_sig[note] = "+"
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return key_sig

    def _calculate_mode_key_sig(self, root: str, mode: str) -> dict[str, str] | None:
        """Calculate key signature for a mode on any root.

        For example, D dorian uses the same notes as C major.
        """
        if mode not in MODE_INTERVALS:
            return None

        # Note to semitone mapping
        note_semitones = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

        # Handle accidentals in root
        root_note = root[0] if root else ""
        if root_note not in note_semitones:
            return None

        root_semitone = note_semitones[root_note]
        if len(root) > 1:
            if root[1] in "#+":
                root_semitone += 1
            elif root[1] in "-b":
                root_semitone -= 1
        root_semitone = root_semitone % 12

        # Calculate the parent major scale
        mode_offset = MODE_INTERVALS[mode]
        parent_semitone = (root_semitone - mode_offset) % 12

        # Find the parent major key
        semitone_to_major = {
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

        parent_major = semitone_to_major.get(parent_semitone)
        if parent_major and parent_major in KEY_SIGNATURES:
            return KEY_SIGNATURES[parent_major].copy()

        return None

    def _process_variable_definition(self, node: VariableDefinitionNode) -> None:
        """Process a variable definition (store only, don't emit sound)."""
        self.state.variables[node.name] = node.events

    def _process_variable_reference(self, node: VariableReferenceNode) -> None:
        """Process a variable reference."""
        if node.name in self.state.variables:
            self._process_event_sequence(self.state.variables[node.name])

    def _process_marker(self, node: MarkerNode) -> None:
        """Process a marker definition."""
        part = self._get_part_state()
        self.state.markers[node.name] = part.current_time

    def _process_at_marker(self, node: AtMarkerNode) -> None:
        """Process a marker reference (jump to marker time)."""
        if node.name in self.state.markers:
            target_time = self.state.markers[node.name]
            for part in self._get_all_part_states():
                part.current_time = target_time

    def _process_voice_group(self, node: VoiceGroupNode) -> None:
        """Process a voice group."""
        all_parts = self._get_all_part_states()
        start_times = {id(p): p.current_time for p in all_parts}
        max_end_time = max(start_times.values())

        for voice in node.voices:
            # Reset to start time for each voice
            for part in all_parts:
                part.current_time = start_times[id(part)]
            self._process_event_sequence(voice.events)
            for part in all_parts:
                max_end_time = max(max_end_time, part.current_time)

        # Advance to the end of the longest voice
        for part in all_parts:
            part.current_time = max_end_time

    def _process_cram(self, node: CramNode) -> None:
        """Process a cram expression."""
        all_parts = self._get_all_part_states()
        part = all_parts[0]  # Use first part for duration calculation

        # Calculate the total duration for the cram
        if node.duration:
            total_beats = self._calculate_duration(node.duration, part)
        else:
            total_beats = part.default_duration

        total_secs = self._beats_to_seconds(total_beats, part.tempo)

        # Count the number of events (notes/rests)
        event_count = self._count_sounding_events(node.events)

        if event_count == 0:
            return

        # Save current state for all parts
        saved_states = {
            id(p): (p.current_time, p.default_duration)
            for p in all_parts
        }

        # Set a temporary duration for each event in all parts
        for p in all_parts:
            p.default_duration = total_beats / event_count

        # Process events
        self._process_event_sequence(node.events)

        # Restore state and set final time for all parts
        for p in all_parts:
            start_time, saved_duration = saved_states[id(p)]
            p.default_duration = saved_duration
            p.current_time = start_time + total_secs

    def _process_repeat(self, node: RepeatNode) -> None:
        """Process a repeat expression."""
        for i in range(node.times):
            self.state.repetition_number = i + 1
            self._process_node(node.event)
        self.state.repetition_number = 1

    def _process_on_repetitions(self, node: OnRepetitionsNode) -> None:
        """Process an on-repetitions expression."""
        # Check if current repetition matches any of the ranges
        current_rep = self.state.repetition_number
        should_play = False

        for r in node.ranges:
            if r.last is None:
                # Single number
                if current_rep == r.first:
                    should_play = True
                    break
            else:
                # Range
                if r.first <= current_rep <= r.last:
                    should_play = True
                    break

        if should_play:
            self._process_node(node.event)

    def _calculate_duration(
        self, duration: DurationNode | None, part: PartState
    ) -> float:
        """Calculate duration in beats from a DurationNode.

        Args:
            duration: The duration node, or None for default duration.
            part: The current part state.

        Returns:
            Duration in beats.
        """
        if duration is None:
            return part.default_duration

        total_beats = 0.0

        for component in duration.components:
            if isinstance(component, NoteLengthNode):
                # Calculate base duration (4 = quarter note = 1 beat)
                beats = 4.0 / component.denominator

                # Apply dots
                dot_value = beats
                for _ in range(component.dots):
                    dot_value /= 2
                    beats += dot_value

                total_beats += beats

            elif isinstance(component, NoteLengthMsNode):
                # Convert ms to beats
                ms = component.ms
                beats_per_second = part.tempo / 60.0
                total_beats += (ms / 1000.0) * beats_per_second

            elif isinstance(component, NoteLengthSecondsNode):
                # Convert seconds to beats
                beats_per_second = part.tempo / 60.0
                total_beats += component.seconds * beats_per_second

        return total_beats

    def _beats_to_seconds(self, beats: float, tempo: float) -> float:
        """Convert beats to seconds.

        Args:
            beats: Number of beats.
            tempo: Tempo in BPM.

        Returns:
            Duration in seconds.
        """
        return beats * 60.0 / tempo

    def _count_sounding_events(self, sequence: EventSequenceNode) -> int:
        """Count the number of note/rest events in a sequence."""
        count = 0
        for event in sequence.events:
            if isinstance(event, (NoteNode, RestNode)):
                count += 1
            elif isinstance(event, ChordNode):
                count += 1  # Chord counts as one event
            elif isinstance(event, CramNode):
                count += 1  # Cram counts as one event
            elif isinstance(event, BracketedSequenceNode):
                count += self._count_sounding_events(event.events)
            elif isinstance(event, RepeatNode):
                inner = 1
                if isinstance(event.event, BracketedSequenceNode):
                    inner = self._count_sounding_events(event.event.events)
                count += inner * event.times
        return count


def generate_midi(ast: RootNode) -> MidiSequence:
    """Convenience function to generate MIDI from an AST.

    Args:
        ast: The root node of the Alda AST.

    Returns:
        A MidiSequence containing all MIDI events.
    """
    generator = MidiGenerator()
    return generator.generate(ast)
