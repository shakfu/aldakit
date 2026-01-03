#include <libremidi/libremidi.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_libremidi, m) {
  // Expose available_apis for debugging
  m.def("available_apis", []() {
    std::vector<std::string> result;
    for (auto api : libremidi::available_apis()) {
      result.push_back(std::string(libremidi::get_api_name(api)));
    }
    return result;
  });

  m.def("get_version", []() {
    return std::string(libremidi::get_version());
  });
  // Error type returned by libremidi functions
  nb::class_<stdx::error>(m, "Error")
      .def("__bool__", [](stdx::error e) { return e != stdx::error{}; })
      .def("__str__", [](stdx::error e) { return e.message().data(); })
      .def("__repr__", [](stdx::error e) { return e.message().data(); });

  nb::class_<libremidi::port_information>(m, "PortInformation")
      .def(nb::init<>())
      .def_rw("client", &libremidi::port_information::client)
      .def_rw("port", &libremidi::port_information::port)
      .def_rw("manufacturer", &libremidi::port_information::manufacturer)
      .def_rw("device_name", &libremidi::port_information::device_name)
      .def_rw("port_name", &libremidi::port_information::port_name)
      .def_rw("display_name", &libremidi::port_information::display_name);

  nb::class_<libremidi::input_port, libremidi::port_information>(m, "InputPort")
      .def(nb::init<>());

  nb::class_<libremidi::output_port, libremidi::port_information>(m, "OutputPort")
      .def(nb::init<>());

  nb::class_<libremidi::observer>(m, "Observer")
      .def("__init__", [](libremidi::observer *self) {
        libremidi::observer_configuration conf;
        conf.track_hardware = true;
        conf.track_virtual = true;  // Enable virtual/software ports
        conf.track_any = true;      // Track any other port types
        new (self) libremidi::observer{std::move(conf), libremidi::midi1::default_api()};
      })
      .def("get_input_ports", &libremidi::observer::get_input_ports)
      .def("get_output_ports", &libremidi::observer::get_output_ports)
      .def("get_current_api", [](libremidi::observer &self) {
        return std::string(libremidi::get_api_name(self.get_current_api()));
      });

  nb::class_<libremidi::midi_out>(m, "MidiOut")
      .def("__init__", [](libremidi::midi_out *self) {
        new (self) libremidi::midi_out{{}, libremidi::midi1::default_api()};
      })
      .def("open_port",
           [](libremidi::midi_out &self, const libremidi::output_port &p) {
             return self.open_port(p);
           })
      .def("open_virtual_port",
           [](libremidi::midi_out &self, std::string_view name) {
             return self.open_virtual_port(name);
           })
      .def("close_port", &libremidi::midi_out::close_port)
      .def("is_port_open", &libremidi::midi_out::is_port_open)
      .def("send_message",
           [](libremidi::midi_out &self, unsigned char b0, unsigned char b1,
              unsigned char b2) { return self.send_message(b0, b1, b2); })
      .def("send_message",
           [](libremidi::midi_out &self, unsigned char b0, unsigned char b1) {
             return self.send_message(b0, b1);
           });
}
