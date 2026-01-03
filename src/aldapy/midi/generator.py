"""MIDI generator that converts an Alda AST to MIDI events."""

from dataclasses import dataclass, field

from ..ast_nodes import (
    ASTNode,
    RootNode,
    PartNode,
    EventSequenceNode,
    NoteNode,
    RestNode,
    ChordNode,
    DurationNode,
    NoteLengthNode,
    NoteLengthMsNode,
    NoteLengthSecondsNode,
    BarlineNode,
    OctaveSetNode,
    OctaveUpNode,
    OctaveDownNode,
    LispListNode,
    LispSymbolNode,
    LispNumberNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    MarkerNode,
    AtMarkerNode,
    VoiceGroupNode,
    CramNode,
    RepeatNode,
    OnRepetitionsNode,
    BracketedSequenceNode,
)
from ..midi.types import (
    MidiSequence,
    MidiNote,
    MidiProgramChange,
    MidiTempoChange,
    INSTRUMENT_PROGRAMS,
    note_to_midi,
)


@dataclass
class PartState:
    """State for a single part/instrument."""

    octave: int = 4
    tempo: float = 120.0  # BPM
    volume: int = 80  # 0-127
    quantization: float = 0.9  # 0.0-1.0, affects note duration
    default_duration: float = 0.25  # Duration in beats (quarter note)
    current_time: float = 0.0  # Current time in seconds
    channel: int = 0
    program: int = 0


@dataclass
class GeneratorState:
    """Global state for the MIDI generator."""

    global_tempo: float = 120.0
    variables: dict[str, EventSequenceNode] = field(default_factory=dict)
    markers: dict[str, float] = field(default_factory=dict)  # marker -> time in seconds
    parts: dict[str, PartState] = field(default_factory=dict)
    current_part: str | None = None
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
        """Get the current part state, creating default if needed."""
        if self.state.current_part is None:
            # Create implicit part
            self.state.current_part = "_default"
            self.state.parts["_default"] = PartState(
                channel=self.state.next_channel,
                program=0,
            )
            self.state.next_channel = min(15, self.state.next_channel + 1)

        return self.state.parts[self.state.current_part]

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
            self._get_part_state().octave = node.octave
        elif isinstance(node, OctaveUpNode):
            self._get_part_state().octave += 1
        elif isinstance(node, OctaveDownNode):
            self._get_part_state().octave -= 1
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

        # Use alias as part name if available, otherwise first instrument name
        part_name = alias if alias else names[0]

        # Create or get part state
        if part_name not in self.state.parts:
            # Determine MIDI program from instrument name
            program = 0
            for name in names:
                normalized = name.lower().replace("_", "-")
                if normalized in INSTRUMENT_PROGRAMS:
                    program = INSTRUMENT_PROGRAMS[normalized]
                    break

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

        self.state.current_part = part_name

        # Process events
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
        part = self._get_part_state()

        # Calculate MIDI note number
        midi_note = note_to_midi(node.letter, part.octave, node.accidentals)

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
        part = self._get_part_state()

        duration_beats = self._calculate_duration(node.duration, part)
        duration_secs = self._beats_to_seconds(duration_beats, part.tempo)

        # Update default duration if specified
        if node.duration is not None:
            part.default_duration = duration_beats

        # Advance time
        part.current_time += duration_secs

    def _process_chord(self, node: ChordNode) -> None:
        """Process a chord (simultaneous notes)."""
        part = self._get_part_state()
        start_time = part.current_time
        max_duration = 0.0

        for item in node.notes:
            if isinstance(item, NoteNode):
                duration = self._process_note(item, is_chord=True)
                max_duration = max(max_duration, duration)
            elif isinstance(item, OctaveSetNode):
                part.octave = item.octave
            elif isinstance(item, OctaveUpNode):
                part.octave += 1
            elif isinstance(item, OctaveDownNode):
                part.octave -= 1
            elif isinstance(item, LispListNode):
                self._process_lisp_list(item)

        # Advance time by the longest note
        part.current_time = start_time + max_duration

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

        part = self._get_part_state()

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
                    part.tempo = new_tempo
                self.sequence.tempo_changes.append(
                    MidiTempoChange(bpm=new_tempo, time=part.current_time)
                )

        elif func_name in ("vol", "volume", "vol!", "volume!"):
            # Set volume (0-100 -> 0-127)
            if args and isinstance(args[0], LispNumberNode):
                vol = int(args[0].value)
                part.volume = min(127, max(0, int(vol * 127 / 100)))

        elif func_name in ("quant", "quantize", "quantization"):
            # Set quantization (0-100 -> 0.0-1.0)
            if args and isinstance(args[0], LispNumberNode):
                quant = float(args[0].value)
                part.quantization = max(0.0, min(1.0, quant / 100.0))

        elif func_name == "panning":
            # Set panning (0-100 -> 0-127)
            if args and isinstance(args[0], LispNumberNode):
                pan = int(args[0].value)
                pan_value = min(127, max(0, int(pan * 127 / 100)))
                from .types import MidiControlChange

                self.sequence.control_changes.append(
                    MidiControlChange(
                        control=10,  # Pan control
                        value=pan_value,
                        time=part.current_time,
                        channel=part.channel,
                    )
                )

        elif func_name in ("octave", "octave!"):
            # Set octave
            if args and isinstance(args[0], LispNumberNode):
                part.octave = int(args[0].value)

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
            dynamics = {
                "pppppp": 10,
                "ppppp": 20,
                "pppp": 30,
                "ppp": 40,
                "pp": 50,
                "p": 60,
                "mp": 70,
                "mf": 80,
                "f": 90,
                "ff": 100,
                "fff": 110,
                "ffff": 115,
                "fffff": 120,
                "ffffff": 127,
            }
            part.volume = dynamics.get(func_name, 80)

    def _process_variable_definition(self, node: VariableDefinitionNode) -> None:
        """Process a variable definition."""
        self.state.variables[node.name] = node.events
        # Also process the events
        self._process_event_sequence(node.events)

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
            part = self._get_part_state()
            part.current_time = self.state.markers[node.name]

    def _process_voice_group(self, node: VoiceGroupNode) -> None:
        """Process a voice group."""
        part = self._get_part_state()
        start_time = part.current_time
        max_end_time = start_time

        for voice in node.voices:
            # Reset to start time for each voice
            part.current_time = start_time
            self._process_event_sequence(voice.events)
            max_end_time = max(max_end_time, part.current_time)

        # Advance to the end of the longest voice
        part.current_time = max_end_time

    def _process_cram(self, node: CramNode) -> None:
        """Process a cram expression."""
        part = self._get_part_state()

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

        # Save current state
        start_time = part.current_time
        saved_duration = part.default_duration

        # Set a temporary duration for each event
        part.default_duration = total_beats / event_count

        # Process events
        self._process_event_sequence(node.events)

        # Restore state and set final time
        part.default_duration = saved_duration
        part.current_time = start_time + total_secs

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
