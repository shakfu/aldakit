#include <libremidi/libremidi.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <deque>
#include <mutex>

namespace nb = nanobind;

// Thread-safe message queue for MIDI input
struct MidiMessage {
    std::vector<unsigned char> bytes;
    int64_t timestamp;
};

// Wrapper for MidiIn that handles callbacks safely with Python GIL
struct MidiInWrapper {
    std::deque<MidiMessage> message_queue;
    std::mutex queue_mutex;
    libremidi::midi_in impl;
    std::function<void(const std::vector<unsigned char>&, int64_t)>
        python_callback;

    MidiInWrapper()
        : impl { create_config(), libremidi::midi1::default_api() }
    {
    }

    libremidi::input_configuration create_config()
    {
        libremidi::input_configuration conf;
        conf.on_message = [this](libremidi::message&& msg) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            message_queue.push_back(MidiMessage {
                std::vector<unsigned char>(msg.begin(), msg.end()),
                msg.timestamp });
        };
        conf.ignore_sysex = true;
        conf.ignore_timing = true;
        conf.ignore_sensing = true;
        return conf;
    }

    stdx::error open_port(const libremidi::input_port& p)
    {
        return impl.open_port(p);
    }

    stdx::error open_virtual_port(std::string_view name)
    {
        return impl.open_virtual_port(name);
    }

    void close_port() { impl.close_port(); }

    bool is_port_open() const { return impl.is_port_open(); }

    // Poll for messages and invoke callback if set, return list of messages
    std::vector<MidiMessage> poll()
    {
        std::vector<MidiMessage> messages;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            while (!message_queue.empty()) {
                messages.push_back(std::move(message_queue.front()));
                message_queue.pop_front();
            }
        }
        return messages;
    }

    // Check if there are pending messages
    bool has_messages()
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return !message_queue.empty();
    }

    // Get current timestamp
    int64_t absolute_timestamp() { return impl.absolute_timestamp(); }
};

NB_MODULE(_libremidi, m)
{
    // Expose available_apis for debugging
    m.def("available_apis", []() {
        std::vector<std::string> result;
        for (auto api : libremidi::available_apis()) {
            result.push_back(std::string(libremidi::get_api_name(api)));
        }
        return result;
    });

    m.def("get_version",
          []() { return std::string(libremidi::get_version()); });

    // Error type returned by libremidi functions
    nb::class_<stdx::error>(m, "Error")
        .def("__bool__", [](stdx::error e) { return e != stdx::error {}; })
        .def("__str__", [](stdx::error e) { return e.message().data(); })
        .def("__repr__", [](stdx::error e) { return e.message().data(); });

    // MIDI message type for input
    nb::class_<MidiMessage>(m, "MidiMessage")
        .def(nb::init<>())
        .def_rw("bytes", &MidiMessage::bytes)
        .def_rw("timestamp", &MidiMessage::timestamp)
        .def("__repr__", [](const MidiMessage& msg) {
            std::string result = std::to_string(msg.timestamp) + ": [";
            for (size_t i = 0; i < msg.bytes.size(); ++i) {
                if (i > 0)
                    result += " ";
                result += std::to_string(static_cast<int>(msg.bytes[i]));
            }
            result += "]";
            return result;
        });

    nb::class_<libremidi::port_information>(m, "PortInformation")
        .def(nb::init<>())
        .def_rw("client", &libremidi::port_information::client)
        .def_rw("port", &libremidi::port_information::port)
        .def_rw("manufacturer", &libremidi::port_information::manufacturer)
        .def_rw("device_name", &libremidi::port_information::device_name)
        .def_rw("port_name", &libremidi::port_information::port_name)
        .def_rw("display_name", &libremidi::port_information::display_name);

    nb::class_<libremidi::input_port, libremidi::port_information>(m,
                                                                   "InputPort")
        .def(nb::init<>());

    nb::class_<libremidi::output_port, libremidi::port_information>(
        m, "OutputPort")
        .def(nb::init<>());

    nb::class_<libremidi::observer>(m, "Observer")
        .def("__init__",
             [](libremidi::observer* self) {
                 libremidi::observer_configuration conf;
                 conf.track_hardware = true;
                 conf.track_virtual = true; // Enable virtual/software ports
                 conf.track_any = true;     // Track any other port types
                 new (self)
                     libremidi::observer { std::move(conf),
                                           libremidi::midi1::default_api() };
             })
        .def("get_input_ports", &libremidi::observer::get_input_ports)
        .def("get_output_ports", &libremidi::observer::get_output_ports)
        .def("get_current_api", [](libremidi::observer& self) {
            return std::string(
                libremidi::get_api_name(self.get_current_api()));
        });

    // MIDI Input with thread-safe message queue
    nb::class_<MidiInWrapper>(m, "MidiIn")
        .def(nb::init<>())
        .def("open_port", &MidiInWrapper::open_port)
        .def("open_virtual_port", &MidiInWrapper::open_virtual_port)
        .def("close_port", &MidiInWrapper::close_port)
        .def("is_port_open", &MidiInWrapper::is_port_open)
        .def("poll", &MidiInWrapper::poll,
             "Poll for incoming MIDI messages. Returns a list of MidiMessage "
             "objects.")
        .def("has_messages", &MidiInWrapper::has_messages,
             "Check if there are pending messages without consuming them.")
        .def("absolute_timestamp", &MidiInWrapper::absolute_timestamp,
             "Get the current absolute timestamp in nanoseconds.");

    nb::class_<libremidi::midi_out>(m, "MidiOut")
        .def("__init__",
             [](libremidi::midi_out* self) {
                 new (self)
                     libremidi::midi_out { {},
                                           libremidi::midi1::default_api() };
             })
        .def("open_port",
             [](libremidi::midi_out& self, const libremidi::output_port& p) {
                 return self.open_port(p);
             })
        .def("open_virtual_port",
             [](libremidi::midi_out& self, std::string_view name) {
                 return self.open_virtual_port(name);
             })
        .def("close_port", &libremidi::midi_out::close_port)
        .def("is_port_open", &libremidi::midi_out::is_port_open)
        .def("send_message",
             [](libremidi::midi_out& self, unsigned char b0, unsigned char b1,
                unsigned char b2) { return self.send_message(b0, b1, b2); })
        .def("send_message",
             [](libremidi::midi_out& self, unsigned char b0,
                unsigned char b1) { return self.send_message(b0, b1); });
}
