"""MIDI generator that converts an Alda AST to MIDI events."""

import math
from dataclasses import dataclass, field, replace

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
from ..errors import AldaGenerationError
from ..constants import (
    BEATS_PER_WHOLE_NOTE,
    DEFAULT_DURATION,
    DEFAULT_OCTAVE,
    DEFAULT_PAN,
    DEFAULT_QUANTIZATION,
    DEFAULT_TEMPO,
    DEFAULT_TRACK_VOLUME,
    DEFAULT_VOLUME,
    DYNAMICS_VELOCITY,
    MIDI_CC_EXPRESSION,
    MIDI_CC_PAN,
    MIDI_DRUM_CHANNEL,
    MIDI_MAX_CHANNELS,
    MIDI_MAX_CONTROL_VALUE,
    MIDI_MAX_NOTE,
    MIDI_MIN_NOTE,
    MILLISECONDS_PER_SECOND,
    SECONDS_PER_MINUTE,
)
from ..midi.types import (
    MidiControlChange,
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    MidiTempoChange,
    is_percussion,
    lookup_instrument,
    note_to_midi_raw,
)
from ..theory import (
    key_signature_from_accidental_words,
    key_signature_from_string,
    key_signature_from_symbols,
)
from .channels import (
    MELODIC_CHANNELS as MELODIC_CHANNELS,
)
from .channels import (
    VIRTUAL_CHANNEL_BASE,
    ChannelAssignment,
    assign_channels,
)


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


def _percent_to_midi(percent: float) -> int:
    """A 0-100 attribute value as a 0-127 MIDI value, rounded as Alda rounds.

    Alda scales to a fraction, multiplies by 127 and rounds half away from
    zero, so panning 50 is 64. Truncating would give 63.
    """
    scaled = math.floor(percent / 100 * MIDI_MAX_CONTROL_VALUE + 0.5)
    return min(MIDI_MAX_CONTROL_VALUE, max(0, scaled))


def _copied(value: object) -> object:
    """A copy of a mutable attribute value, so parts do not share one dict."""
    return dict(value) if isinstance(value, dict) else value


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
    quantization: float = DEFAULT_QUANTIZATION  # Fraction of duration sounded; may exceed 1
    default_duration: float = DEFAULT_DURATION  # Beats (quarter note = 1 beat)
    current_time: float = 0.0  # Current time in seconds
    # Factor applied to note and rest lengths inside a cram expression
    time_scale: float = 1.0
    #: The MIDI channel this part sounds on. While the AST is being walked
    #: this is a placeholder; generate() replaces it with a real channel once
    #: the score's shape is known. -1 means the part never sounds, so it needs
    #: no channel at all.
    channel: int = 0
    #: The placeholder the part's events were emitted on, kept so that the
    #: linter can attribute a shared channel back to the parts sharing it.
    allocated_channel: int = 0
    program: int = 0
    key_signature: dict[str, str] = field(default_factory=dict)  # note -> accidental
    transpose: int = 0  # Transposition in semitones
    percussion: bool = False  # True for midi-percussion (channel 9)
    pan: int = DEFAULT_PAN  # MIDI CC 10, 0-127
    track_volume: int = DEFAULT_TRACK_VOLUME  # MIDI CC 11, 0-127


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
    next_channel: int = 0  # Number of virtual channels handed out so far
    # The virtual channels handed out, in order. Channels are assigned for
    # real once the whole score is known; see aldakit.midi.channels.
    allocated_channels: list[int] = field(default_factory=list)
    repetition_number: int = 1  # Current repetition when in a repeat loop
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # Aliased instrument groups: alias -> {instrument name: internal part name}.
    # Populated by 'violin/viola "strings":' so that 'strings.viola:' resolves.
    groups: dict[str, dict[str, str]] = field(default_factory=dict)
    # Instances of each instrument, keyed by program ("percussion" for drums):
    # whether an unnamed one exists, and whether a named one does. Alda
    # refuses a score that refers to both.
    unnamed_instances: set[object] = field(default_factory=set)
    named_instances: set[object] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    # Each note with the program, pan and track volume its part had when the
    # note was generated: (note, program or None for percussion, pan, volume).
    note_settings: list[tuple[MidiNote, int | None, int, int]] = field(
        default_factory=list
    )
    # Where each voice group ended: (part's channel, index of the part's first
    # note after the group). See _separate_voice_group_overlaps.
    voice_group_ends: list[tuple[int, int]] = field(default_factory=list)
    # The MIDI tempo map, built as Alda builds it: the first declared part's
    # tempo changes, overridden by global (tempo!) changes at the same time.
    # Other parts' local tempos only place their own notes. Keyed by time.
    master_part: str | None = None  # Name of the first declared part
    master_tempos: dict[float, float] = field(default_factory=dict)
    global_tempos: dict[float, float] = field(default_factory=dict)


class MidiGenerator(ASTVisitor):
    """Generates MIDI events from an Alda AST."""

    def __init__(self, strict: bool = False) -> None:
        """
        Args:
            strict: Raise AldaGenerationError on the first diagnostic instead
                of reporting it and continuing. Alda stops on these errors.
        """
        self.strict = strict
        self.sequence = MidiSequence()
        self.state = GeneratorState()
        self.channel_assignment = ChannelAssignment()

    def generate(self, ast: RootNode) -> MidiSequence:
        """Generate a MIDI sequence from an Alda AST.

        Args:
            ast: The root node of the Alda AST.

        Returns:
            A MidiSequence containing all MIDI events.
        """
        self.sequence = MidiSequence()
        self.state = GeneratorState()
        self.channel_assignment = ChannelAssignment()

        # Process all children
        for child in ast.children:
            self.visit(child)

        self._separate_voice_group_overlaps()
        self._emit_channel_settings()

        # Turn the parts' placeholder channels into real MIDI channels, now
        # that the score's shape is known.
        self.channel_assignment = assign_channels(
            self.sequence, self.state.allocated_channels, self._warn
        )
        self._resolve_part_channels()

        tempos = {0.0: float(DEFAULT_TEMPO)}
        tempos.update(self.state.master_tempos)
        tempos.update(self.state.global_tempos)
        self.sequence.tempo_changes = [
            MidiTempoChange(bpm=bpm, time=time) for time, bpm in tempos.items()
        ]

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
        """Record a non-fatal problem, or raise it when generation is strict."""
        diagnostic = Diagnostic(message, position, code)
        if self.strict:
            raise AldaGenerationError(diagnostic)
        self.state.diagnostics.append(diagnostic)

    def _allocate_channel(self) -> int:
        """Allocate a virtual channel for a melodic part.

        Which real channel a part sounds on depends on how many parts the
        score turns out to have and on when each of them is playing, neither
        of which is known while the AST is still being walked. Parts are given
        a placeholder here and :func:`aldakit.midi.channels.assign_channels`
        rewrites it at the end of generation.
        """
        channel = VIRTUAL_CHANNEL_BASE + self.state.next_channel
        self.state.next_channel += 1
        self.state.allocated_channels.append(channel)
        return channel

    def _resolve_part_channels(self) -> None:
        """Replace each part's placeholder channel with the real one.

        A part that reuses more than one channel over the course of the score
        reports the first; the full picture is in ``channel_assignment``.
        """
        assigned = self.channel_assignment.channels
        for part in self.state.parts.values():
            part.allocated_channel = part.channel
            if part.channel not in assigned:
                continue  # percussion, or pinned with (midi-channel)
            channels = assigned[part.channel]
            part.channel = channels[0] if channels else -1

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
                channel=self._allocate_channel(), program=0
            )
            self.state.master_part = self.state.master_part or "_default"

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

        self._check_instances(names, alias, node.declaration.position)

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

                self.state.parts[part_name] = self._new_part_state(
                    channel=self._allocate_channel(), program=program
                )

            active_parts.append(part_name)

        if self.state.master_part is None and active_parts:
            self.state.master_part = active_parts[0]

        # Record group membership so "alias.instrument" can address one member
        if alias:
            self.state.groups[alias] = group_members

        self.state.current_parts = active_parts

        # Process events (will be applied to all active parts)
        self.visit(node.events)

    def _check_instances(self, names: list[str], alias: str | None, position) -> None:
        """Report a score that refers to unnamed and named instances of one
        instrument, which Alda refuses as ambiguous (doc/instance-and-group-
        assignment.md). An aliased group is exempt when created, as in Alda.
        """

        def key(name: str) -> object:
            return "percussion" if is_percussion(name) else lookup_instrument(name)

        stock = [
            n for n in names if n not in self.state.aliases and key(n) is not None
        ]
        if alias is None:
            for name in stock:
                if key(name) in self.state.named_instances:
                    self._ambiguous(name, position)
                self.state.unnamed_instances.add(key(name))
            return
        if len(names) == 1 and stock and key(stock[0]) in self.state.unnamed_instances:
            self._ambiguous(stock[0], position)
        self.state.named_instances.update(key(n) for n in stock)
        self.state.aliases.add(alias)

    def _ambiguous(self, name: str, position) -> None:
        self._warn(
            f"Ambiguous instrument reference {name!r}: the score uses both "
            "named and unnamed instances of it. Name every instance.",
            position,
            code="ambiguous-instance",
        )

    def visit_EventSequenceNode(self, node: EventSequenceNode) -> None:
        """Process a sequence of events."""
        for event in node.events:
            self.visit(event)

    def _process_note(self, node: NoteNode, is_chord: bool = False) -> dict[int, float]:
        """Process a note, returning its duration in seconds for each part.

        Active parts can differ in tempo and default duration, so one note
        written once is a different number of seconds long in each of them.

        Args:
            node: The note node.
            is_chord: If True, don't advance time after the note.

        Returns:
            Duration in seconds keyed by ``id()`` of the part state.
        """
        durations: dict[int, float] = {}

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
            duration_secs = (
                self._beats_to_seconds(duration_beats, part.tempo) * part.time_scale
            )

            # Apply quantization (affects actual note length, not timing)
            if node.slurred:
                actual_duration = duration_secs  # Full duration for slurred notes
            else:
                actual_duration = duration_secs * part.quantization

            midi_note_event = MidiNote(
                pitch=midi_note,
                velocity=part.volume,
                start_time=part.current_time,
                duration=actual_duration,
                channel=part.channel,
            )
            self.sequence.notes.append(midi_note_event)
            self.state.note_settings.append(
                (
                    midi_note_event,
                    None if part.percussion else part.program,
                    part.pan,
                    part.track_volume,
                )
            )

            # Update default duration if specified
            if node.duration is not None:
                part.default_duration = duration_beats

            durations[id(part)] = duration_secs

            # Advance time (unless in chord)
            if not is_chord:
                part.current_time += duration_secs

        return durations

    def visit_RestNode(self, node: RestNode) -> None:
        self._process_rest(node)

    def _process_rest(self, node: RestNode, is_chord: bool = False) -> dict[int, float]:
        """Process a rest, returning its duration in seconds for each part.

        Args:
            node: The rest node.
            is_chord: If True, don't advance time after the rest.
        """
        durations: dict[int, float] = {}
        for part in self._get_all_part_states():
            duration_beats = self._calculate_duration(node.duration, part)
            if node.duration is not None:
                part.default_duration = duration_beats
            durations[id(part)] = (
                self._beats_to_seconds(duration_beats, part.tempo) * part.time_scale
            )
            if not is_chord:
                part.current_time += durations[id(part)]
        return durations

    def visit_ChordNode(self, node: ChordNode) -> None:
        """Process a chord (simultaneous notes)."""
        # Save start times for all active parts
        all_parts = self._get_all_part_states()
        start_times = {id(p): p.current_time for p in all_parts}
        # The next event follows the chord's shortest note or rest
        # (docs/alda-language/chords.md). The length is per part: in
        # violin/viola: the two parts can be at different tempi.
        shortest: dict[int, float] = {}

        for item in node.notes:
            if isinstance(item, NoteNode):
                durations = self._process_note(item, is_chord=True)
            elif isinstance(item, RestNode):
                durations = self._process_rest(item, is_chord=True)
            else:
                self.visit(item)
                continue
            for part_id, duration in durations.items():
                shortest[part_id] = min(shortest.get(part_id, duration), duration)

        for part in all_parts:
            part.current_time = start_times[id(part)] + shortest.get(id(part), 0.0)

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

    def _checked_arg(
        self,
        func_name: str,
        args: list,
        minimum: float = 0.0,
        maximum: float | None = None,
        positive: bool = False,
    ) -> float | None:
        """The first argument as a number, or None if missing or out of range.

        The ranges are the ones Alda enforces (client/model/lisp.go). A value
        outside them is reported and ignored.
        """
        value = self._number_arg(args)
        if value is None:
            return None
        if positive and value <= minimum:
            requirement = "a positive number"
        elif value < minimum:
            requirement = "a non-negative number"
        elif maximum is not None and value > maximum:
            requirement = f"between {minimum:g} and {maximum:g}"
        else:
            return value
        self._warn(
            f"({func_name} {value:g}): the value must be {requirement}; ignored.",
            args[0].position,
            code="invalid-attribute-value",
        )
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
        return state

    @handles("tempo", "tempo!")
    def _set_tempo(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set the tempo in beats per minute."""
        new_tempo = self._checked_arg(func_name, args, positive=True)
        if new_tempo is None:
            return
        if func_name == "tempo!":
            self.state.global_tempo = new_tempo
        self._set_attribute(func_name, parts, "tempo", new_tempo)
        if func_name == "tempo!":
            time = parts[0].current_time if parts else 0.0
            self.state.global_tempos[round(time, 9)] = new_tempo
            return
        master = self.state.parts.get(self.state.master_part or "")
        if master is not None and any(part is master for part in parts):
            self.state.master_tempos[round(master.current_time, 9)] = new_tempo

    @handles("vol", "volume", "vol!", "volume!")
    def _set_volume(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set volume on Alda's 0-100 scale, stored as MIDI velocity."""
        vol = self._checked_arg(func_name, args, maximum=100)
        if vol is None:
            return
        self._set_attribute(func_name, parts, "volume", _percent_to_midi(vol))

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
        quant = self._checked_arg(func_name, args)
        if quant is None:
            return
        # No upper bound: Alda accepts quant above 100, which holds each note
        # past the start of the next.
        self._set_attribute(func_name, parts, "quantization", max(0.0, quant / 100.0))

    @handles("panning", "pan", "panning!", "pan!")
    def _set_panning(self, func_name: str, args: list, parts: list[PartState]) -> None:
        """Set the pan, sent as MIDI CC 10 with the part's next note."""
        pan = self._checked_arg(func_name, args, maximum=100)
        if pan is None:
            return
        self._set_attribute(func_name, parts, "pan", _percent_to_midi(pan))

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
        beats = self._checked_arg(func_name, args, positive=True)
        if beats is None:
            return
        self._set_attribute(func_name, parts, "default_duration", beats)

    @handles("set-note-length", "set-note-length!")
    def _set_note_length(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set the default note length as a note value, e.g. 1 for a whole note."""
        denominator = self._checked_arg(func_name, args, positive=True)
        if denominator is None:
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
        ms = self._checked_arg(func_name, args, positive=True)
        if ms is None:
            return
        for part in self._target_parts(func_name, parts):
            beats_per_second = part.tempo / SECONDS_PER_MINUTE
            part.default_duration = (ms / MILLISECONDS_PER_SECOND) * beats_per_second

    @handles("track-volume", "track-vol", "track-volume!", "track-vol!")
    def _set_track_volume(
        self, func_name: str, args: list, parts: list[PartState]
    ) -> None:
        """Set Alda's track-volume, sent as MIDI CC 11 (expression) as Alda does.

        This is the instrument's overall level, as opposed to ``volume``, which
        is the velocity of individual notes.
        """
        level = self._checked_arg(func_name, args, maximum=100)
        if level is None:
            return
        self._set_attribute(func_name, parts, "track_volume", _percent_to_midi(level))

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
            # The part's next note selects its instrument on the new channel
            part.channel = channel

    def _separate_voice_group_overlaps(self) -> None:
        """Move a part to a new channel after a voice group, when it must.

        Alda moves a part to a new channel after every voice group. aldakit
        keeps the channel unless a note from before the group's end is still
        sounding when the part plays the same pitch: on one channel the new
        note would cut the old one off, where Alda sounds both
        (docs/dev/alda-deviations.md, D1).
        """
        moved: dict[int, int] = {}  # Virtual channel -> where its later notes went
        notes = self.sequence.notes
        for channel, index in self.state.voice_group_ends:
            while channel in moved:
                channel = moved[channel]
            later = [n for n in notes[index:] if n.channel == channel]
            if not later:
                continue
            first = min(n.start_time for n in later)
            sounding = [
                n
                for n in notes[:index]
                if n.channel == channel and n.start_time + n.duration > first + 1e-9
            ]
            if not any(
                e.pitch == n.pitch
                and e.start_time <= n.start_time < e.start_time + e.duration - 1e-9
                for e in sounding
                for n in later
            ):
                continue
            new = self._allocate_channel()
            for n in later:
                n.channel = new
            moved[channel] = new

    def _emit_channel_settings(self) -> None:
        """Send the program, pan and track volume each note needs.

        Walking the notes in time order, each setting is sent at a note's start
        when it differs from what the note's channel last received. This is
        what Alda does. It keeps parts pinned to one channel, and voices of one
        part, from overriding each other's settings.
        """
        sent: dict[int, dict[str, int]] = {}
        for note, program, pan, volume in sorted(
            self.state.note_settings, key=lambda s: s[0].start_time
        ):
            channel = sent.setdefault(note.channel, {})
            if program is not None and channel.get("program") != program:
                self.sequence.program_changes.append(
                    MidiProgramChange(
                        program=program, time=note.start_time, channel=note.channel
                    )
                )
                channel["program"] = program
            for name, control, value in (
                ("pan", MIDI_CC_PAN, pan),
                ("track_volume", MIDI_CC_EXPRESSION, volume),
            ):
                if channel.get(name) != value:
                    self.sequence.control_changes.append(
                        MidiControlChange(
                            control=control,
                            value=value,
                            time=note.start_time,
                            channel=note.channel,
                        )
                    )
                    channel[name] = value

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
        nested = False
        i = 0
        while i < len(node.elements):
            elem = node.elements[i]
            if isinstance(elem, LispSymbolNode):
                symbols.append(elem.name.lower())
            elif isinstance(elem, LispListNode):
                # Nested list like (flat) or (sharp)
                nested = True
                if elem.elements and isinstance(elem.elements[0], LispSymbolNode):
                    symbols.append(elem.elements[0].name.lower())
            i += 1

        if nested:
            # A nested (flat) or (sharp) marks the association-list form. The
            # symbols alone could read as a key name once flattened, so the
            # structure is what settles it.
            return key_signature_from_accidental_words(symbols)
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
        """Fork each active part into one copy per voice, then merge them.

        As in Alda (client/model/voice.go), every voice starts from the part's
        state at the start of the group, and a voice number used again
        continues where that voice left off. At the end, the voice that
        finished last becomes the part, with all of its state; ties go to the
        voice created last.
        """
        self._get_part_state()  # an implicit part, if none is active yet
        names = list(self.state.current_parts)
        templates = {name: self.state.parts[name] for name in names}
        voices: dict[str, dict[int, PartState]] = {name: {} for name in names}

        for voice in node.voices:
            for name in names:
                if voice.number not in voices[name]:
                    template = templates[name]
                    voices[name][voice.number] = replace(
                        template, key_signature=dict(template.key_signature)
                    )
                self.state.parts[name] = voices[name][voice.number]
            self.state.current_parts = list(names)
            self.visit(voice.events)

        for name in names:
            forks = list(voices[name].values())
            if not forks:
                continue
            winner = forks[-1]
            for fork in forks[:-1]:
                if fork.current_time > winner.current_time:
                    winner = fork
            self.state.parts[name] = winner
            if not winner.percussion and winner.channel >= VIRTUAL_CHANNEL_BASE:
                self.state.voice_group_ends.append(
                    (winner.channel, len(self.sequence.notes))
                )
        self.state.current_parts = names

    def visit_CramNode(self, node: CramNode) -> None:
        """Fit the cram's events into its duration, keeping their proportions.

        Each part's events are scaled by the cram's duration divided by the
        sum of the events' own durations, as Alda does (client/model/cram.go).
        A nested cram multiplies the scales.
        """
        all_parts = self._get_all_part_states()
        saved = {id(p): (p.default_duration, p.time_scale) for p in all_parts}
        outer_beats: dict[int, float] = {}
        for p in all_parts:
            outer_beats[id(p)] = self._calculate_duration(node.duration, p)
            inner = self._inner_seconds(
                node.events, p.default_duration, p.tempo, self.state.repetition_number
            )[0]
            if inner <= 0:
                return  # nothing in the cram takes time
            outer = self._beats_to_seconds(outer_beats[id(p)], p.tempo)
            p.time_scale = p.time_scale * outer / inner

        self.visit(node.events)

        for p in all_parts:
            default_duration, time_scale = saved[id(p)]
            p.time_scale = time_scale
            # A cram's own duration becomes the default for what follows
            if node.duration is not None:
                p.default_duration = outer_beats[id(p)]
            else:
                p.default_duration = default_duration

    def _inner_seconds(
        self, node, default_duration: float, tempo: float, repetition: int
    ) -> tuple[float, float]:
        """The unscaled length of ``node`` and the default duration after it.

        Mirrors Alda's DurationMs: note and rest lengths carry over as defaults,
        a chord counts its shortest note, a nested cram its own duration, and
        attributes nothing.
        """
        part = PartState(tempo=tempo, default_duration=default_duration)

        def secs(duration) -> float:
            beats = self._calculate_duration(duration, part)
            if duration is not None:
                part.default_duration = beats
            return self._beats_to_seconds(beats, tempo)

        def walk(node, repetition: int) -> float:
            if isinstance(node, (NoteNode, RestNode)):
                return secs(node.duration)
            if isinstance(node, CramNode):
                return self._beats_to_seconds(
                    self._calculate_duration(node.duration, part), tempo
                )
            if isinstance(node, ChordNode):
                lengths = [walk(n, repetition) for n in node.notes]
                return min((x for x in lengths if x > 0), default=0.0)
            if isinstance(node, (EventSequenceNode,)):
                return sum(walk(e, repetition) for e in node.events)
            if isinstance(node, BracketedSequenceNode):
                return walk(node.events, repetition)
            if isinstance(node, RepeatNode):
                return sum(walk(node.event, i + 1) for i in range(node.times))
            if isinstance(node, OnRepetitionsNode):
                applies = any(
                    r.first == repetition
                    if r.last is None
                    else r.first <= repetition <= r.last
                    for r in node.ranges
                )
                return walk(node.event, repetition) if applies else 0.0
            if isinstance(node, VariableReferenceNode):
                events = self.state.variables.get(node.name)
                return walk(events, repetition) if events is not None else 0.0
            return 0.0

        return walk(node, repetition), part.default_duration

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


def generate_midi(ast: RootNode, strict: bool = False) -> MidiSequence:
    """Convenience function to generate MIDI from an AST.

    Args:
        ast: The root node of the Alda AST.
        strict: Raise on the first diagnostic; see MidiGenerator.

    Returns:
        A MidiSequence containing all MIDI events.

    Raises:
        AldaGenerationError: If ``strict`` and the score has a diagnostic.
    """
    generator = MidiGenerator(strict=strict)
    return generator.generate(ast)
