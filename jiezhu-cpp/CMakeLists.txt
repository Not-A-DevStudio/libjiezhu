cmake_minimum_required(VERSION 3.20)
project(jiezhu LANGUAGES CXX VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# When this project is built standalone (not via the top-level), keep
# build artifacts (dll/lib/a) in the same folder as executables.
set(_JIEZHU_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_JIEZHU_OUTPUT_DIR}")
endif ()
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_JIEZHU_OUTPUT_DIR}")
endif ()
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${_JIEZHU_OUTPUT_DIR}")
endif ()

if (CMAKE_CONFIGURATION_TYPES)
    foreach (_cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${_cfg}" _cfg_u)
        if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_u})
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_u} "${_JIEZHU_OUTPUT_DIR}/${_cfg}")
        endif ()
        if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_u})
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_u} "${_JIEZHU_OUTPUT_DIR}/${_cfg}")
        endif ()
        if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg_u})
            set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg_u} "${_JIEZHU_OUTPUT_DIR}/${_cfg}")
        endif ()
    endforeach ()
endif ()

include(GNUInstallDirs)

# ========= begin Jiezhu build options =========
option(JIEZHU_BUILD_STATIC "Set ON to build Jiezhu " OFF)
option(JIEZHU_INSTALL "Install jiezhu CMake package" ON)
option(JIEZHU_ENABLE_JIEZHU_ABILITY "Enable full Jiezhu ability in the library" ON)
option(JIEZHU_BUILD_TEST "build Jiezhu test" OFF)
# =========  end Jiezhu build options  =========

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/chat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/jiezhu.cpp
)
if (JIEZHU_ENABLE_JIEZHU_ABILITY)
    set(JIEZHU_LIB_NAME "jiezhu_full")
else ()
    set(JIEZHU_LIB_NAME "jiezhu")
endif ()

if (JIEZHU_BUILD_STATIC)
    add_library(${JIEZHU_LIB_NAME} STATIC ${SOURCES})
    message("JIEZHU: Build static")
else ()
    add_library(${JIEZHU_LIB_NAME} SHARED ${SOURCES})
    message("JIEZHU: Build dynamic")
endif ()

# Export friendly alias targets for consumers
add_library(jiezhu::jie ALIAS ${JIEZHU_LIB_NAME})
add_library(jiezhu::jiezhu ALIAS ${JIEZHU_LIB_NAME})

# Ensure bundled dependencies participate in install/export when packaging.
# Otherwise, exporting jiezhu would fail because it links to non-exported targets.
if (JIEZHU_INSTALL)
    set(JSON_Install ON CACHE BOOL "Install nlohmann_json when building as a subproject" FORCE)
endif ()

add_subdirectory(third_party)

# Prefer the canonical CURL::libcurl target for exported usage requirements.
# When curl is built as a subproject, it may only define `libcurl`.
if (TARGET CURL::libcurl)
    set(_JIEZHU_CURL_TARGET CURL::libcurl)
elseif (TARGET libcurl)
    add_library(CURL::libcurl ALIAS libcurl)
    set(_JIEZHU_CURL_TARGET CURL::libcurl)
else ()
    message(FATAL_ERROR "jiezhu requires a curl target (CURL::libcurl or libcurl) but none was found")
endif ()

target_include_directories(${JIEZHU_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(${JIEZHU_LIB_NAME} PUBLIC ${_JIEZHU_CURL_TARGET} nlohmann_json::nlohmann_json)

if (JIEZHU_ENABLE_JIEZHU_ABILITY)
    target_compile_definitions(${JIEZHU_LIB_NAME} PUBLIC JIE_ENABLE_JIEZHU_ABLITY)
    message("JIEZHU: Enable full Jiezhu ability")
endif ()

if (JIEZHU_BUILD_TEST)
    message("JIEZHU: Build tests")
    include(CTest)
    if (BUILD_TESTING)
        add_executable(jiezhu-tests
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_chat_curl.cpp
        )
        target_link_libraries(jiezhu-tests PRIVATE jiezhu Catch2::Catch2WithMain)
        if (WIN32)
            target_link_libraries(jiezhu-tests PRIVATE ws2_32)
        endif ()

        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Catch2-3.12.0/extras")
        include(Catch)
        catch_discover_tests(jiezhu-tests)
    endif ()
endif ()
if (JIEZHU_INSTALL)
    include(CMakePackageConfigHelpers)

    install(
            TARGETS ${JIEZHU_LIB_NAME}
            EXPORT jiezhuTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
            DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    set(_JIEZHU_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

    install(
            EXPORT jiezhuTargets
            FILE jiezhuTargets.cmake
            NAMESPACE jiezhu::
            DESTINATION ${_JIEZHU_CMAKE_INSTALL_DIR}
    )

    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jiezhuConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/jiezhuConfig.cmake
            INSTALL_DESTINATION ${_JIEZHU_CMAKE_INSTALL_DIR}
            NO_SET_AND_CHECK_MACRO
            NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )
    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/jiezhuConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion
    )

    install(
            FILES
            ${CMAKE_CURRENT_BINARY_DIR}/jiezhuConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/jiezhuConfigVersion.cmake
            DESTINATION ${_JIEZHU_CMAKE_INSTALL_DIR}
    )
endif ()
