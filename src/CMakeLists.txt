# _libremidi nanobind extension module
nanobind_add_module(_libremidi aldapy/_libremidi.cpp)
target_link_libraries(_libremidi PUBLIC libremidi)

set_target_properties(_libremidi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src/aldapy"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src/aldapy"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src/aldapy"
)


# Install directive for scikit-build-core
install(TARGETS _libremidi LIBRARY DESTINATION aldapy)
