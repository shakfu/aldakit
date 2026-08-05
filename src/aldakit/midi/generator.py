"""MIDI generator that converts an Alda AST to MIDI events."""

from dataclasses import dataclass, field

from ..ast_nodes import (
    ASTVisitor,
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
    PartDeclarationNode,
    PartNode,
    RepeatNode,
    RestNode,
    RootNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    VoiceGroupNode,
)
from ..constants import (
    BEATS_PER_WHOLE_NOTE,
    DEFAULT_DURATION,
    DEFAULT_OCTAVE,
    DEFAULT_QUANTIZATION,
    DEFAULT_TEMPO,
    DEFAULT_VOLUME,
    DYNAMICS_VELOCITY,
    MIDI_CC_PAN,
    MIDI_CC_VOLUME,
    MIDI_DRUM_CHANNEL,
    MIDI_MAX_CHANNELS,
    MIDI_MAX_CONTROL_VALUE,
    MIDI_MAX_NOTE,
    MIDI_MAX_VELOCITY,
    MIDI_MIN_NOTE,
    MIDI_MIN_VELOCITY,
    MILLISECONDS_PER_SECOND,
    SECONDS_PER_MINUTE,
)
from ..midi.types import (
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    MidiTempoChange,
    is_percussion,
    lookup_instrument,
    note_to_midi_raw,
)
from ..theory import key_signature_from_string, key_signature_from_symbols


# Attribute name as written in a score -> name of the MidiGenerator method
# that applies it. Populated by the @handles decorator on those methods, so
# adding an attribute means writing one method, not editing a dispatch chain.
ATTRIBUTE_HANDLERS: dict[str, str] = {}


def handles(*names: str):
    """Register the decorated method as the handler for these attributes.

    Handlers take ``(func_name, args, parts)``: the attribute as written (so
    a handler can tell ``tempo`` from ``tempo!``), its unevaluated arguments,
    and the part states currently active.
    """

    def decorate(method):
        for name in names:
            ATTRIBUTE_HANDLERS[name] = method.__name__
        return method

    return decorate


def _copied(value: object) -> object:
    """A copy of a mutable attribute value, so parts do not share one dict."""
    return dict(value) if isinstance(value, dict) else value


# Channels available to pitched instruments. Channel 9 (MIDI channel 10) is
# reserved by the General MIDI spec for percussion, where the note number
# selects a drum sound rather than a pitch.
MELODIC_CHANNELS: tuple[int, ...] = tuple(
    c for c in range(MIDI_MAX_CHANNELS) if c != MIDI_DRUM_CHANNEL
)


@dataclass
class Diagnostic:
    """A non-fatal problem found while generating MIDI."""

    message: str
    position: object = None  # SourcePosition | None
    #: Short stable slug for the kind of problem, e.g. "unknown-instrument".
    #: Lets tools group and filter diagnostics without matching on prose.
    code: str = ""

    def __str__(self) -> str:
        if self.position is not None:
            return f"{self.position}: {self.message}"
        return self.message


@dataclass
class PartState:
    """State for a single part/instrument."""

    octave: int = DEFAULT_OCTAVE
    tempo: float = float(DEFAULT_TEMPO)  # BPM
    volume: int = DEFAULT_VOLUME  # 0-127, default mf (54% of 127)
    quantization: float = DEFAULT_QUANTIZATION  # 0.0-1.0, affects note duration
    default_duration: float = DEFAULT_DURATION  # Beats (quarter note = 1 beat)
    current_time: float = 0.0  # Current time in seconds
    channel: int = 0
    program: int = 0
    key_signature: dict[str, str] = field(default_factory=dict)  # note -> accidental
    transpose: int = 0  # Transposition in semitones
    percussion: bool = False  # True for midi-percussion (channel 9)
    # Channel volume (MIDI CC 7), 0-127. None until the score sets it, so a
    # score that never mentions track-volume emits no CC 7 at all.
    track_volume: int | None = None


@dataclass
class GeneratorState:
    """Global state for the MIDI generator."""

    global_tempo: float = float(DEFAULT_TEMPO)
    variables: dict[str, EventSequenceNode] = field(default_factory=dict)
    markers: dict[str, float] = field(default_factory=dict)  # marker -> time in seconds
    parts: dict[str, PartState] = field(default_factory=dict)
    current_parts: list[str] = field(
        default_factory=list
    )  # Active parts (multi-instrument support)
    # Attribute values set globally with a trailing "!", keyed by the
    # PartState field they set. Parts declared after the fact start with them,
    # which is what makes a global attribute at the top of a score apply to
    # the whole score.
    global_attributes: dict[str, object] = field(default_factory=dict)
    next_channel: int = 0  # Index into MELODIC_CHANNELS
    repetition_number: int = 1  # Current repetition when in a repeat loop
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # Aliased instrument groups: alias -> {instrument name: internal part name}.
    # Populated by 'violin/viola "strings":' so that 'strings.viola:' resolves.
    groups: dict[str, dict[str, str]] = field(default_factory=dict)


class MidiGenerator(ASTVisitor):
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
            self.visit(child)

        # Sort events by time
        self.sequence.notes.sort(key=lambda n: n.start_time)
        self.sequence.program_changes.sort(key=lambda p: p.time)
        self.sequence.tempo_changes.sort(key=lambda t: t.time)

        return self.sequence

    @property
    def diagnostics(self) -> list[Diagnostic]:
        """Non-fatal problems found during the last generate() call.

        Includes unknown instrument names, undefined variable and marker
        references, and MIDI channel exhaustion.
        """
        return self.state.diagnostics

    def _warn(self, message: str, position: object = None, code: str = "") -> None:
        """Record a non-fatal problem."""
        self.state.diagnostics.append(Diagnostic(message, position, code))

    def _allocate_channel(self) -> int:
        """Allocate the next melodic MIDI channel, skipping the drum channel.

        There are only 15 melodic channels. Once they are exhausted, channels
        are reused from the start and a diagnostic is recorded, because the
        reused channel's program change will be overwritten.
        """
        index = self.state.next_channel
        if index >= len(MELODIC_CHANNELS):
            self._warn(
                f"More than {len(MELODIC_CHANNELS)} melodic parts declared; "
                "MIDI channels are being reused and instrument assignments "
                "will collide.",
                code="channel-exhaustion",
            )
            index %= len(MELODIC_CHANNELS)
        self.state.next_channel += 1
        return MELODIC_CHANNELS[index]

    def _resolve_group_member(self, name: str) -> str | None:
        """Resolve a dotted group-member reference such as ``strings.cello``.

        Args:
            name: The reference as written in the score.

        Returns:
            The internal part name, or None if this is not a resolvable
            group-member reference.
        """
        if "." not in name:
            return None
        group, _, member = name.partition(".")
        members = self.state.groups.get(group)
        if members is None:
            return None
        return members.get(member.lower())

    def _get_part_state(self) -> PartState:
        """Get the current part state (first active part), creating default if needed."""
        if not self.state.current_parts:
            # Create implicit part
            self.state.current_parts = ["_default"]
            self.state.parts["_default"] = self._new_part_state(
                channel=self._allocate_channel(),
                program=0,
            )

        return self.state.parts[self.state.current_parts[0]]

    def _get_all_part_states(self) -> list[PartState]:
        """Get all currently active part states."""
        if not self.state.current_parts:
            return [self._get_part_state()]
        return [self.state.parts[name] for name in self.state.current_parts]

    def visit_OctaveSetNode(self, node: OctaveSetNode) -> None:
        for part in self._get_all_part_states():
            part.octave = node.octave

    def visit_OctaveUpNode(self, node: OctaveUpNode) -> None:
        for part in self._get_all_part_states():
            part.octave += 1

    def visit_OctaveDownNode(self, node: OctaveDownNode) -> None:
        for part in self._get_all_part_states():
            part.octave -= 1

    def visit_BarlineNode(self, node: BarlineNode) -> None:
        pass  # Barlines are purely visual

    def visit_BracketedSequenceNode(self, node: BracketedSequenceNode) -> None:
        self.visit(node.events)

    def visit_NoteNode(self, node: NoteNode) -> None:
        self._process_note(node)

    def visit_PartDeclarationNode(self, node: PartDeclarationNode) -> None:
        """Handle a declaration that is not wrapped in a PartNode.

        It switches the active part; the events that follow are siblings.
        """
        self.visit_PartNode(
            PartNode(
                declaration=node,
                events=EventSequenceNode(events=[], position=node.position),
                position=node.position,
            )
        )

    def visit_PartNode(self, node: PartNode) -> None:
        """Process a part declaration and its events."""
        # Get instrument name(s)
        names = node.declaration.names
        alias = node.declaration.alias

        # A dotted reference selects one member of a previously aliased group
        # (violin/viola/cello "strings":  then  strings.cello:).
        if len(names) == 1 and "." in names[0] and alias is None:
            resolved = self._resolve_group_member(names[0])
            if resolved is not None:
                self.state.current_parts = [resolved]
                self.visit(node.events)
                return
            self._warn(
                f"Unknown group member {names[0]!r}; no aliased group defines it.",
                node.declaration.position,
                code="unknown-group-member",
            )

        # For multi-instrument parts (violin/viola/cello), create a part for each
        # The alias applies to the group but each instrument gets its own channel
        active_parts = []
        group_members: dict[str, str] = {}

        for i, name in enumerate(names):
            # Use alias+index for group naming, or just instrument name
            if alias and len(names) > 1:
                part_name = f"{alias}_{i}"
            elif alias:
                part_name = alias
            else:
                part_name = name

            group_members[name.lower()] = part_name

            # Create or get part state
            if part_name not in self.state.parts:
                if is_percussion(name):
                    # Percussion always lives on the GM drum channel and takes
                    # no program change: the note number selects the drum sound.
                    self.state.parts[part_name] = self._new_part_state(
                        channel=MIDI_DRUM_CHANNEL,
                        program=0,
                        percussion=True,
                    )
                    active_parts.append(part_name)
                    continue

                # Determine MIDI program from instrument name
                program = lookup_instrument(name)
                if program is None:
                    if "." not in name:
                        # A dotted name already reported an unresolved group
                        self._warn(
                            f"Unknown instrument {name!r}; "
                            "falling back to acoustic grand piano.",
                            node.declaration.position,
                            code="unknown-instrument",
                        )
                    program = 0

                channel = self._allocate_channel()

                self.state.parts[part_name] = self._new_part_state(
                    channel=channel,
                    program=program,
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

        # Record group membership so "alias.instrument" can address one member
        if alias:
            self.state.groups[alias] = group_members

        self.state.current_parts = active_parts

        # Process events (will be applied to all active parts)
        self.visit(node.events)

    def visit_EventSequenceNode(self, node: EventSequenceNode) -> None:
        """Process a sequence of events."""
        for event in node.events:
            self.visit(event)

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
            if part.percussion:
                # On the drum channel a note number names a drum, so key
                # signatures and transposition must not shift it.
                accidentals = node.accidentals
            elif not accidentals:
                # No explicit accidentals - check key signature
                letter = node.letter.lower()
                if letter in part.key_signature:
                    accidentals = [part.key_signature[letter]]
            elif "_" in accidentals:
                # Natural sign explicitly cancels key signature
                accidentals = []

            # Calculate the MIDI note number, including transposition, and
            # report rather than silently clamp a note outside 0-127.
            raw_pitch = note_to_midi_raw(node.letter, part.octave, accidentals)
            if part.transpose != 0 and not part.percussion:
                raw_pitch += part.transpose
            midi_note = max(MIDI_MIN_NOTE, min(MIDI_MAX_NOTE, raw_pitch))
            if raw_pitch != midi_note:
                self._warn(
                    f"Note {node.letter!r} in octave {part.octave} is outside "
                    f"the MIDI range ({raw_pitch}); clamped to {midi_note}.",
                    node.position,
                    code="note-out-of-range",
                )

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

    def visit_RestNode(self, node: RestNode) -> None:
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

    def visit_ChordNode(self, node: ChordNode) -> None:
        """Process a chord (simultaneous notes)."""
        # Save start times for all active parts
        all_parts = self._get_all_part_states()
        start_times = {id(p): p.current_time for p in all_parts}
        max_duration = 0.0

        for item in node.notes:
            if isinstance(item, NoteNode):
                duration = self._process_note(item, is_chord=True)
                max_duration = max(max_duration, duration)
            else:
                self.visit(item)

        # Advance time by the longest note for all parts
        for part in all_parts:
            part.current_time = start_times[id(part)] + max_duration

    def visit_LispListNode(self, node: LispListNode) -> None:
        """Apply an attribute S-expression such as ``(tempo 120)``."""
        if not node.elements:
            return

        first = node.elements[0]
        if not isinstance(first, LispSymbolNode):
            return

        func_name = first.name.lower()
        args = node.elements[1:]

        handler_name = ATTRIBUTE_HANDLERS.get(func_name)
        if handler_name is None:
            self._warn(
                f"Unknown attribute {func_name!r}; ignored.",
                node.position,
                code="unknown-attribute",
            )
            return

        # A global attribute before the first part declaration applies to
        # every part through state.global_attributes, so it must not force an
        # implicit part into existence: that would spend channel 0 on a part
        # with no notes and push the score's first instrument to channel 1.
        if func_name.endswith("!") and not self.state.current_parts:
            parts: list[PartState] = []
        else:
            # All active parts, so 'violin/viola:' sets the attribute on both.
            parts = self._get_all_part_states()

        getattr(self, handler_name)(func_name, args, parts)

    @staticmethod
    def _number_arg(args: list) -> float | None:
        """The first argument as a number, or None if it is not one."""
        if args and isinstance(args[0], LispNumberNode):
            return float(args[0].value)
        return None

    def _target_parts(self, func_name: str, parts: list[PartState]) -> list[PartState]:
        """Parts an attribute applies to.

        A trailing ``!`` makes an attribute global, which in Alda means it
        applies to every part rather than only the ones currently active.
        """
        if func_name.endswith("!"):
            return list(self.state.parts.values())
        return parts

    def _set_attribute(
        self, func_name: str, parts: list[PartState], field_name: str, value: object
    ) -> None:
        """Set a part-state field on the parts an attribute applies to.

        A global attribute is also remembered, so a part declared later in the
        score -- including every part, when the attribute is written above the
        first declaration -- starts out with it.
        """
        if func_name.endswith("!"):
            self.state.global_attributes[field_name] = value
        for part in self._target_parts(func_name, parts):
            setattr(part, field_name, _copied(value))

    def _new_part_state(self, **kwargs) -> PartState:
        """Create a part state carrying the global attributes set so far."""
        state = PartState(tempo=self.state.global_tempo, **kwargs)
        for field_name, value in self.state.global_attributes.items():
            setattr(state, field_name, _copied(value))
        if state.track_volume is not None:
            # Inherited from a global (track-volume! ...): the new part has to
            # send the control change on its own channel.
            self._emit_track_volume(state, 0.0)
        return state

    @handles("tempo", "tempo!")
    def _set_tempo(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set the tempo in beats per minute."""
        new_tempo = self._number_arg(args)
        if new_tempo is None:
            return
        if func_name == "tempo!":
            self.state.global_tempo = new_tempo
        self._set_attribute(func_name, parts, "tempo", new_tempo)
        self.sequence.tempo_changes.append(
            MidiTempoChange(bpm=new_tempo, time=parts[0].current_time if parts else 0.0)
        )

    @handles("vol", "volume", "vol!", "volume!")
    def _set_volume(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set volume on Alda's 0-100 scale, stored as MIDI velocity."""
        vol = self._number_arg(args)
        if vol is None:
            return
        velocity = min(
            MIDI_MAX_VELOCITY,
            max(MIDI_MIN_VELOCITY, int(vol * MIDI_MAX_VELOCITY / 100)),
        )
        self._set_attribute(func_name, parts, "volume", velocity)

    @handles(
        "quant",
        "quantize",
        "quantization",
        "quant!",
        "quantize!",
        "quantization!",
    )
    def _set_quantization(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set the fraction of its duration a note actually sounds for."""
        quant = self._number_arg(args)
        if quant is None:
            return
        quantization = max(0.0, min(1.0, quant / 100.0))
        self._set_attribute(func_name, parts, "quantization", quantization)

    @handles("panning", "pan", "panning!", "pan!")
    def _set_panning(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Emit a pan control change on each active part's channel."""
        pan = self._number_arg(args)
        if pan is None:
            return
        pan_value = min(
            MIDI_MAX_CONTROL_VALUE,
            max(0, int(pan * MIDI_MAX_CONTROL_VALUE / 100)),
        )
        from .types import MidiControlChange

        for part in self._target_parts(func_name, parts):
            self.sequence.control_changes.append(
                MidiControlChange(
                    control=MIDI_CC_PAN,
                    value=pan_value,
                    time=part.current_time,
                    channel=part.channel,
                )
            )

    @handles("octave", "octave!")
    def _set_octave(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set the octave to a number, or shift it with 'up / 'down."""
        if not args:
            return

        target = self._target_parts(func_name, parts)
        octave = self._number_arg(args)
        if octave is not None:
            for part in target:
                part.octave = int(octave)
            return

        # 'up and 'down, quoted as Alda writes them or bare for convenience.
        arg = args[0]
        if isinstance(arg, LispQuotedNode) and isinstance(arg.value, LispSymbolNode):
            symbol = arg.value.name.lower()
        elif isinstance(arg, LispSymbolNode):
            symbol = arg.name.lower()
        else:
            return

        if symbol == "up":
            for part in target:
                part.octave += 1
        elif symbol == "down":
            for part in target:
                part.octave -= 1

    @handles(*DYNAMICS_VELOCITY)
    def _set_dynamic(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Apply a dynamic marking such as (mf) as a volume level."""
        velocity = DYNAMICS_VELOCITY[func_name]
        for part in parts:
            part.volume = velocity

    @handles("key-sig", "key-signature", "key-sig!", "key-signature!")
    def _set_key_signature(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set the key signature applied to unaltered notes."""
        key_sig = self._parse_key_signature(args)
        if key_sig is None:
            return
        self._set_attribute(func_name, parts, "key_signature", key_sig)

    @handles("transpose", "transpose!", "transposition", "transposition!")
    def _set_transposition(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Shift every subsequent note by a number of semitones."""
        semitones = self._number_arg(args)
        if semitones is None:
            return
        self._set_attribute(func_name, parts, "transpose", int(semitones))

    @handles("set-duration", "set-duration!")
    def _set_duration(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set the default note length in beats, e.g. 2.5 for a dotted half."""
        beats = self._number_arg(args)
        if beats is None or beats <= 0:
            return
        self._set_attribute(func_name, parts, "default_duration", beats)

    @handles("set-note-length", "set-note-length!")
    def _set_note_length(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set the default note length as a note value, e.g. 1 for a whole note."""
        denominator = self._number_arg(args)
        if denominator is None or denominator <= 0:
            return
        self._set_attribute(
            func_name, parts, "default_duration", BEATS_PER_WHOLE_NOTE / denominator
        )

    @handles("set-duration-ms", "set-duration-ms!")
    def _set_duration_ms(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set the default note length in milliseconds.

        Milliseconds are converted to beats per part, because parts can be at
        different tempos.
        """
        ms = self._number_arg(args)
        if ms is None or ms < 0:
            return
        for part in self._target_parts(func_name, parts):
            beats_per_second = part.tempo / SECONDS_PER_MINUTE
            part.default_duration = (ms / MILLISECONDS_PER_SECOND) * beats_per_second

    @handles("track-volume", "track-vol", "track-volume!", "track-vol!")
    def _set_track_volume(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set the channel volume (MIDI CC 7), Alda's track-volume.

        This is the instrument's overall level, as opposed to ``volume``, which
        is the velocity of individual notes.
        """
        level = self._number_arg(args)
        if level is None:
            return
        value = min(
            MIDI_MAX_CONTROL_VALUE,
            max(0, int(level * MIDI_MAX_CONTROL_VALUE / 100)),
        )
        self._set_attribute(func_name, parts, "track_volume", value)
        for part in self._target_parts(func_name, parts):
            self._emit_track_volume(part, part.current_time)

    @handles("midi-channel")
    def _set_midi_channel(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Pin a part to a specific MIDI channel.

        Channel 9 is the General MIDI drum channel, so a melodic part asking
        for it is reported and left where it is rather than silently turning
        into drum hits.
        """
        channel = self._number_arg(args)
        if channel is None:
            return
        channel = int(channel)
        if not 0 <= channel < MIDI_MAX_CHANNELS:
            self._warn(
                f"MIDI channel {channel} is outside 0-{MIDI_MAX_CHANNELS - 1}; "
                "ignored.",
                code="invalid-midi-channel",
            )
            return

        for part in parts:
            if channel == MIDI_DRUM_CHANNEL and not part.percussion:
                self._warn(
                    f"Channel {MIDI_DRUM_CHANNEL} is reserved for percussion; "
                    "ignoring (midi-channel 9) in a melodic part.",
                    code="invalid-midi-channel",
                )
                continue
            if part.channel == channel:
                continue
            previous = part.channel
            part.channel = channel
            self._release_channel(part, previous)
            # The instrument has to be selected again on the new channel.
            if not part.percussion:
                self.sequence.program_changes.append(
                    MidiProgramChange(
                        program=part.program,
                        time=part.current_time,
                        channel=channel,
                    )
                )
            if part.track_volume is not None:
                self._emit_track_volume(part, part.current_time)

    def _release_channel(self, part: PartState, channel: int) -> None:
        """Undo the program change for a channel a part left without using.

        A part is given a channel when it is declared, so ``(midi-channel N)``
        as the part's first event leaves a program change on a channel that
        never sounds a note. Exported files show that as an empty track with
        an instrument on it, so drop it.
        """
        if any(note.channel == channel for note in self.sequence.notes):
            return
        if any(
            other.channel == channel
            for other in self.state.parts.values()
            if other is not part
        ):
            return
        self.sequence.program_changes = [
            pc for pc in self.sequence.program_changes if pc.channel != channel
        ]

    def _emit_track_volume(self, part: PartState, time: float) -> None:
        """Emit the channel-volume control change for a part."""
        if part.track_volume is None:
            return
        from .types import MidiControlChange

        self.sequence.control_changes.append(
            MidiControlChange(
                control=MIDI_CC_VOLUME,
                value=part.track_volume,
                time=time,
                channel=part.channel,
            )
        )

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
            return key_signature_from_string(arg.value)

        # Quoted list format: '(g minor)
        if isinstance(arg, LispQuotedNode):
            return self._parse_key_sig_quoted(arg.value)

        return None

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

        return key_signature_from_symbols(symbols)

    def visit_VariableDefinitionNode(self, node: VariableDefinitionNode) -> None:
        """Process a variable definition (store only, don't emit sound)."""
        self.state.variables[node.name] = node.events

    def visit_VariableReferenceNode(self, node: VariableReferenceNode) -> None:
        """Process a variable reference."""
        if node.name in self.state.variables:
            self.visit(self.state.variables[node.name])
        else:
            self._warn(
                f"Undefined variable {node.name!r}.",
                node.position,
                code="undefined-variable",
            )

    def visit_MarkerNode(self, node: MarkerNode) -> None:
        """Process a marker definition."""
        part = self._get_part_state()
        self.state.markers[node.name] = part.current_time

    def visit_AtMarkerNode(self, node: AtMarkerNode) -> None:
        """Process a marker reference (jump to marker time)."""
        if node.name in self.state.markers:
            target_time = self.state.markers[node.name]
            for part in self._get_all_part_states():
                part.current_time = target_time
        else:
            self._warn(
                f"Undefined marker {node.name!r}.",
                node.position,
                code="undefined-marker",
            )

    def visit_VoiceGroupNode(self, node: VoiceGroupNode) -> None:
        """Process a voice group."""
        all_parts = self._get_all_part_states()
        start_times = {id(p): p.current_time for p in all_parts}
        max_end_time = max(start_times.values())

        for voice in node.voices:
            # Reset to start time for each voice
            for part in all_parts:
                part.current_time = start_times[id(part)]
            self.visit(voice.events)
            for part in all_parts:
                max_end_time = max(max_end_time, part.current_time)

        # Advance to the end of the longest voice
        for part in all_parts:
            part.current_time = max_end_time

    def visit_CramNode(self, node: CramNode) -> None:
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
        saved_states = {id(p): (p.current_time, p.default_duration) for p in all_parts}

        # Set a temporary duration for each event in all parts
        for p in all_parts:
            p.default_duration = total_beats / event_count

        # Process events
        self.visit(node.events)

        # Restore state and set final time for all parts
        for p in all_parts:
            start_time, saved_duration = saved_states[id(p)]
            p.default_duration = saved_duration
            p.current_time = start_time + total_secs

    def visit_RepeatNode(self, node: RepeatNode) -> None:
        """Process a repeat expression."""
        for i in range(node.times):
            self.state.repetition_number = i + 1
            self.visit(node.event)
        self.state.repetition_number = 1

    def visit_OnRepetitionsNode(self, node: OnRepetitionsNode) -> None:
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
            self.visit(node.event)

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
                beats = BEATS_PER_WHOLE_NOTE / component.denominator

                # Apply dots
                dot_value = beats
                for _ in range(component.dots):
                    dot_value /= 2
                    beats += dot_value

                total_beats += beats

            elif isinstance(component, NoteLengthMsNode):
                # Convert ms to beats
                ms = component.ms
                beats_per_second = part.tempo / SECONDS_PER_MINUTE
                total_beats += (ms / MILLISECONDS_PER_SECOND) * beats_per_second

            elif isinstance(component, NoteLengthSecondsNode):
                # Convert seconds to beats
                beats_per_second = part.tempo / SECONDS_PER_MINUTE
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
        return beats * SECONDS_PER_MINUTE / tempo

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
