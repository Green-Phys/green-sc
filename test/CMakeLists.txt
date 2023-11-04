project(sc_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(sc_test sc_test.cpp)
target_link_libraries(sc_test
        PRIVATE
        Catch2::Catch2
        GREEN::H5PP sc)

include(CTest)
include(Catch)
catch_discover_tests(sc_test)
