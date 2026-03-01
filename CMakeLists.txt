cmake_minimum_required(VERSION 3.20)
project(jiezhu_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put executables and libraries into the same folder in the build tree.
# This is especially useful on Windows where running the exe requires the DLL
# to be next to it.
set(_JIEZHU_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_JIEZHU_OUTPUT_DIR}")
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_JIEZHU_OUTPUT_DIR}")
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${_JIEZHU_OUTPUT_DIR}")
endif()

# Multi-config generators (Visual Studio) use per-config output dirs.
if(CMAKE_CONFIGURATION_TYPES)
    foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${_cfg}" _cfg_u)
        if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_u})
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_u} "${_JIEZHU_OUTPUT_DIR}/${_cfg}")
        endif()
        if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_u})
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_u} "${_JIEZHU_OUTPUT_DIR}/${_cfg}")
        endif()
        if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg_u})
            set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg_u} "${_JIEZHU_OUTPUT_DIR}/${_cfg}")
        endif()
    endforeach()
endif()

include(FetchContent)
include(CTest)

FetchContent_Declare(
    jiezhu
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/jiezhu-cpp
)
FetchContent_MakeAvailable(jiezhu)

add_executable(jiezhu-cpp main.cpp)
# You can also use `jiezhu::jiezhu` instead of `jiezhu::jie`.
# However, we recommend using `jiezhu::jie` because headers are located in `jie` folder.
target_link_libraries(jiezhu-cpp PRIVATE jiezhu::jie)