# _libremidi nanobind extension module
nanobind_add_module(_libremidi aldakit/_libremidi.cpp)
target_link_libraries(_libremidi PUBLIC libremidi)

# What goes in the wheel. The build tree is where the artifacts live; they
# used to also be written into src/aldakit, which left one Python version's
# extension sitting in the source directory while another was being built.
install(TARGETS _libremidi LIBRARY DESTINATION aldakit)

# _tsf nanobind extension module (TinySoundFont + miniaudio)
nanobind_add_module(_tsf aldakit/_tsf.cpp)
target_include_directories(_tsf PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/TinySoundFont
    ${CMAKE_SOURCE_DIR}/thirdparty/miniaudio
)

# Platform-specific audio frameworks
if(APPLE)
    target_link_libraries(_tsf PRIVATE
        "-framework CoreFoundation"
        "-framework CoreAudio"
        "-framework AudioToolbox"
    )
elseif(UNIX AND NOT APPLE)
    # Linux: link pthread, dl, m (miniaudio needs these)
    target_link_libraries(_tsf PRIVATE pthread dl m)
elseif(WIN32)
    # Windows: no extra libs needed, miniaudio uses COM
endif()

install(TARGETS _tsf LIBRARY DESTINATION aldakit)
