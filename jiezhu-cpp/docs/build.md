# Build Instructions

This library uses CMake for building and requires:

- CMake >= 3.20
- C++17 compiler

The repository contains two levels of CMake:

- Top-level [CMakeLists.txt](../CMakeLists.txt): includes `cpp/` as a subproject (`FetchContent`)
- C++ library itself [cpp/CMakeLists.txt](../CMakeLists.txt): generates the `jiezhu` library and optionally installs exports

You may also find a **pre-built binary package in the Release page** of this repository.

## 1) Build Only the C++ Library (Recommended)

Run the following commands in the root directory of the repository:
```bash
cmake -S . -B build
cmake --build build --config Release
```

or only for `cpp/`：

```bash
cmake -S cpp -B build-cpp
cmake --build build-cpp --config Release
```

## 2) Add as a Dependency (add_subdirectory)

```cmake
add_subdirectory(path/to/jiezhu/cpp)

target_link_libraries(your_target PRIVATE jiezhu::jie)
```

## 3) Install and find_package

`cpp/CMakeLists.txt` ：

- Options: `JIEZHU_INSTALL` (default `ON`)
- After installation, a CMake package will be generated and installed: `jiezhuConfig.cmake` / `jiezhuTargets.cmake`

Example:

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%CD%/install
cmake --build build-cpp --config Release
cmake --install build-cpp
```

After installation, downstream projects can use `find_package` to locate and link against `jiezhu`:

```cmake
find_package(jiezhu CONFIG REQUIRED)
target_link_libraries(app PRIVATE jiezhu::jie)
```

NOTE: installing Jiezhu is not tested, if you want to use this method, please report any issues you encounter. I will try to fix them as soon as possible.

## 4) CMake Options

- `JIEZHU_BUILD_STATIC` (default `OFF`): Build static (`STATIC`) or shared (`SHARED`) library
- `JIEZHU_INSTALL` (default `ON`): Whether to generate install/export rules
- `JIEZHU_ENABLE_JIEZHU_ABLITY` (default `ON`): Whether to enable full ability of `jiezhu` (if `OFF`, only the core API is built, without functions like `chat_completion_jiezhu`)
- `JIEZHU_BULID_TESTS` (default `OFF`): Whether to build tests (requires `Catch2`)

## 5) Dependency Management

All the third-party dependencies are listed in `cpp/third_party` folder and directly included as subprojects in `cpp/CMakeLists.txt` using `FetchContent`. The dependencies include: 
- libcurl: Prefer using the `CURL::libcurl` target; if the subproject only provides `libcurl`, an alias `CURL::libcurl` will be created
- nlohmann/json: Link to `nlohmann_json::nlohmann_json`

Therefore, for users, simply linking `jiezhu::jie` will provide include paths and dependencies.
