/**
 * TinySoundFont + miniaudio nanobind wrapper for aldakit.
 *
 * Provides direct audio synthesis from MIDI events using SoundFont files.
 */

// Prevent Windows headers from defining min/max macros
#ifdef _WIN32
#define NOMINMAX
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// TinySoundFont implementation
#define TSF_IMPLEMENTATION
#include "tsf.h"

// miniaudio implementation
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING      // We don't need encoding
#define MA_NO_GENERATION    // We don't need generation
#include "miniaudio.h"

#include <atomic>
#include <mutex>
#include <vector>
#include <cstring>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;  // For _a suffix

/**
 * A scheduled MIDI note event.
 */
struct ScheduledNote {
    int channel;
    int key;
    float velocity;      // 0.0 - 1.0
    double start_time;   // seconds
    double end_time;     // seconds
    bool started;
    bool stopped;
};

/**
 * A scheduled program change event.
 */
struct ScheduledProgram {
    int channel;
    int program;
    double time;
    bool applied;
};

/**
 * A scheduled MIDI control change event.
 */
struct ScheduledControl {
    int channel;
    int control;
    int value;
    double time;
    bool applied;
};

/**
 * TinySoundFont player with miniaudio backend.
 *
 * Provides scheduled MIDI playback with direct audio output.
 */
class TsfPlayer {
public:
    static constexpr int SAMPLE_RATE = 44100;
    static constexpr float TAIL_SECONDS = 0.5f;  // Release tail after last note
    static constexpr int DRUM_CHANNEL = 9;       // General MIDI percussion channel

    TsfPlayer()
        : tsf_(nullptr)
        , device_initialized_(false)
        , playing_(false)
        , current_time_(0.0)
        , global_gain_(1.0f)
    {
        std::memset(&device_, 0, sizeof(device_));
    }

    ~TsfPlayer() {
        stop();
        if (device_initialized_) {
            ma_device_uninit(&device_);
        }
        if (tsf_) {
            tsf_close(tsf_);
        }
    }

    // Non-copyable
    TsfPlayer(const TsfPlayer&) = delete;
    TsfPlayer& operator=(const TsfPlayer&) = delete;

    /**
     * Load a SoundFont file.
     */
    bool load_soundfont(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (tsf_) {
            tsf_close(tsf_);
            tsf_ = nullptr;
        }

        tsf_ = tsf_load_filename(path.c_str());
        if (!tsf_) {
            return false;
        }

        // Configure output: stereo interleaved, 44.1kHz
        tsf_set_output(tsf_, TSF_STEREO_INTERLEAVED, SAMPLE_RATE, global_gain_);

        return true;
    }

    /**
     * Check if a SoundFont is loaded.
     */
    bool is_loaded() const {
        return tsf_ != nullptr;
    }

    /**
     * Get the number of presets in the loaded SoundFont.
     */
    int preset_count() const {
        if (!tsf_) return 0;
        return tsf_get_presetcount(tsf_);
    }

    /**
     * Get preset name by index.
     */
    std::string preset_name(int index) const {
        if (!tsf_ || index < 0 || index >= preset_count()) {
            return "";
        }
        const char* name = tsf_get_presetname(tsf_, index);
        return name ? name : "";
    }

    /**
     * Set the global gain (volume).
     */
    void set_gain(float gain) {
        std::lock_guard<std::mutex> lock(mutex_);
        global_gain_ = std::max(0.0f, std::min(2.0f, gain));
        if (tsf_) {
            tsf_set_output(tsf_, TSF_STEREO_INTERLEAVED, SAMPLE_RATE, global_gain_);
        }
    }

    /**
     * Schedule a program change.
     */
    void schedule_program(int channel, int program, double time) {
        std::lock_guard<std::mutex> lock(mutex_);
        ScheduledProgram pc;
        pc.channel = channel;
        pc.program = program;
        pc.time = time;
        pc.applied = false;
        scheduled_programs_.push_back(pc);
    }

    /**
     * Schedule a MIDI control change (pan, volume, expression, ...).
     */
    void schedule_control(int channel, int control, int value, double time) {
        std::lock_guard<std::mutex> lock(mutex_);
        ScheduledControl cc;
        cc.channel = channel;
        cc.control = control;
        cc.value = value;
        cc.time = time;
        cc.applied = false;
        scheduled_controls_.push_back(cc);
    }

    /**
     * Schedule a note.
     */
    void schedule_note(int channel, int key, float velocity,
                       double start_time, double duration) {
        std::lock_guard<std::mutex> lock(mutex_);
        ScheduledNote note;
        note.channel = channel;
        note.key = key;
        note.velocity = std::max(0.0f, std::min(1.0f, velocity));
        note.start_time = start_time;
        note.end_time = start_time + duration;
        note.started = false;
        note.stopped = false;
        scheduled_notes_.push_back(note);
    }

    /**
     * Clear all scheduled events.
     */
    void clear_schedule() {
        std::lock_guard<std::mutex> lock(mutex_);
        scheduled_notes_.clear();
        scheduled_programs_.clear();
        scheduled_controls_.clear();
        current_time_ = 0.0;
    }

    /**
     * Get the total duration of scheduled notes.
     */
    double duration() const {
        if (scheduled_notes_.empty()) return 0.0;
        double max_end = 0.0;
        for (const auto& note : scheduled_notes_) {
            max_end = std::max(max_end, note.end_time);
        }
        return max_end;
    }

    /**
     * Start playback.
     */
    bool play() {
        if (!tsf_) return false;
        if (playing_) return true;

        // Initialize audio device if needed
        if (!device_initialized_) {
            ma_device_config config = ma_device_config_init(ma_device_type_playback);
            config.playback.format = ma_format_f32;
            config.playback.channels = 2;
            config.sampleRate = SAMPLE_RATE;
            config.dataCallback = audio_callback;
            config.pUserData = this;
            config.periodSizeInFrames = 512;  // Low latency

            if (ma_device_init(nullptr, &config, &device_) != MA_SUCCESS) {
                return false;
            }
            device_initialized_ = true;
        }

        // Reset playback state
        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_time_ = 0.0;
            for (auto& note : scheduled_notes_) {
                note.started = false;
                note.stopped = false;
            }
            for (auto& pc : scheduled_programs_) {
                pc.applied = false;
            }
            for (auto& cc : scheduled_controls_) {
                cc.applied = false;
            }
            tsf_reset(tsf_);
            // General MIDI reserves channel 10 (9 zero-indexed) for percussion,
            // where note numbers select drums from bank 128.
            tsf_channel_set_bank_preset(tsf_, DRUM_CHANNEL, 128, 0);
        }

        playing_ = true;

        if (ma_device_start(&device_) != MA_SUCCESS) {
            playing_ = false;
            return false;
        }

        return true;
    }

    /**
     * Stop playback.
     */
    void stop() {
        if (playing_ && device_initialized_) {
            ma_device_stop(&device_);
        }
        playing_ = false;

        std::lock_guard<std::mutex> lock(mutex_);
        if (tsf_) {
            tsf_note_off_all(tsf_);
        }
    }

    /**
     * Check if currently playing.
     */
    bool is_playing() const {
        return playing_;
    }

    /**
     * Get current playback time.
     */
    double current_time() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_time_;
    }

private:
    static void audio_callback(ma_device* device, void* output,
                               const void* /* input */, ma_uint32 frame_count) {
        auto* player = static_cast<TsfPlayer*>(device->pUserData);
        player->render_audio(static_cast<float*>(output), frame_count);
    }

    void render_audio(float* output, ma_uint32 frame_count) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!tsf_ || !playing_) {
            std::memset(output, 0, frame_count * 2 * sizeof(float));
            return;
        }

        const double time_per_sample = 1.0 / static_cast<double>(SAMPLE_RATE);
        const double seq_duration = duration();

        for (ma_uint32 i = 0; i < frame_count; ++i) {
            // Apply program changes
            for (auto& pc : scheduled_programs_) {
                if (!pc.applied && current_time_ >= pc.time) {
                    if (pc.channel != DRUM_CHANNEL) {
                        tsf_channel_set_presetindex(tsf_, pc.channel, pc.program);
                    }
                    pc.applied = true;
                }
            }

            // Apply control changes
            for (auto& cc : scheduled_controls_) {
                if (!cc.applied && current_time_ >= cc.time) {
                    tsf_channel_midi_control(tsf_, cc.channel, cc.control, cc.value);
                    cc.applied = true;
                }
            }

            // Process note events
            for (auto& note : scheduled_notes_) {
                if (!note.started && current_time_ >= note.start_time) {
                    tsf_channel_note_on(tsf_, note.channel, note.key, note.velocity);
                    note.started = true;
                }
                if (!note.stopped && note.started && current_time_ >= note.end_time) {
                    tsf_channel_note_off(tsf_, note.channel, note.key);
                    note.stopped = true;
                }
            }

            // Render one stereo sample
            tsf_render_float(tsf_, output + (i * 2), 1, 0);
            current_time_ += time_per_sample;
        }

        // Check if playback is complete
        if (scheduled_notes_.empty()) {
            // No notes scheduled - stop immediately
            playing_ = false;
        } else if (current_time_ > seq_duration + TAIL_SECONDS) {
            // All notes done + tail time elapsed
            bool all_stopped = std::all_of(
                scheduled_notes_.begin(), scheduled_notes_.end(),
                [](const ScheduledNote& n) { return n.stopped; }
            );
            if (all_stopped) {
                playing_ = false;
            }
        }
    }

    tsf* tsf_;
    ma_device device_;
    bool device_initialized_;
    std::atomic<bool> playing_;
    double current_time_;
    float global_gain_;

    std::vector<ScheduledNote> scheduled_notes_;
    std::vector<ScheduledProgram> scheduled_programs_;
    std::vector<ScheduledControl> scheduled_controls_;
    mutable std::mutex mutex_;
};


NB_MODULE(_tsf, m) {
    m.doc() = "TinySoundFont audio synthesis backend for aldakit";

    nb::class_<TsfPlayer>(m, "TsfPlayer",
        "SoundFont synthesizer with scheduled MIDI playback.")
        .def(nb::init<>())
        .def("load_soundfont", &TsfPlayer::load_soundfont,
             "path"_a,
             "Load a SoundFont file (.sf2). Returns True on success.")
        .def("is_loaded", &TsfPlayer::is_loaded,
             "Check if a SoundFont is loaded.")
        .def("preset_count", &TsfPlayer::preset_count,
             "Get the number of presets in the loaded SoundFont.")
        .def("preset_name", &TsfPlayer::preset_name,
             "index"_a,
             "Get the name of a preset by index.")
        .def("set_gain", &TsfPlayer::set_gain,
             "gain"_a,
             "Set global gain (0.0 - 2.0, default 1.0).")
        .def("schedule_program", &TsfPlayer::schedule_program,
             "channel"_a, "program"_a, "time"_a,
             "Schedule a program change.")
        .def("schedule_control", &TsfPlayer::schedule_control,
             "channel"_a, "control"_a, "value"_a, "time"_a,
             "Schedule a MIDI control change (e.g. control 10 for pan)")
        .def("schedule_note", &TsfPlayer::schedule_note,
             "channel"_a, "key"_a, "velocity"_a, "start_time"_a, "duration"_a,
             "Schedule a note (velocity 0.0-1.0, times in seconds).")
        .def("clear_schedule", &TsfPlayer::clear_schedule,
             "Clear all scheduled events.")
        .def("duration", &TsfPlayer::duration,
             "Get total duration of scheduled notes in seconds.")
        .def("play", &TsfPlayer::play,
             "Start playback. Returns True on success.")
        .def("stop", &TsfPlayer::stop,
             "Stop playback.")
        .def("is_playing", &TsfPlayer::is_playing,
             "Check if currently playing.")
        .def("current_time", &TsfPlayer::current_time,
             "Get current playback position in seconds.");
}
