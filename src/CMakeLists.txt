#
# Library for self-consistency procedure
#

project(SClib)

add_library(sc INTERFACE)

target_link_libraries(sc INTERFACE GREEN::H5PP GREEN::NDARRAY GREEN::PARAMS GREEN::UTILS GREEN::GRIDS GREEN::OPT)
target_include_directories(sc INTERFACE .)
