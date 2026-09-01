cmake_minimum_required(VERSION 3.20)
project(CuteGenOracle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
  add_compile_options(/permissive- /Zc:__cplusplus /Zc:preprocessor
                      /Zc:externConstexpr /Zc:inline)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(WIN32)
  find_package(Python3 REQUIRED COMPONENTS Development.Module)
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import nanobind, os; print(os.path.dirname(nanobind.__file__))"
  RESULT_VARIABLE _nanobind_status
  OUTPUT_VARIABLE NANOBIND_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _nanobind_error)
if(NOT _nanobind_status EQUAL 0)
  message(FATAL_ERROR
    "nanobind is not installed in ${Python3_EXECUTABLE}: ${_nanobind_error}")
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_path('include'))"
  RESULT_VARIABLE _python_include_status
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _python_include_status EQUAL 0)
  message(FATAL_ERROR "could not determine Python include directory")
endif()

set(CUTEGEN_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/cutlass_compiler/cutegen/include")
set(NANOBIND_COMBINED "${NANOBIND_ROOT}/src/nb_combined.cpp")
if(NOT EXISTS "${NANOBIND_COMBINED}")
  message(FATAL_ERROR "nanobind combined source was not found: ${NANOBIND_COMBINED}")
endif()

add_library(_cutegen_oracle MODULE binding.cpp ${NANOBIND_COMBINED})
target_include_directories(_cutegen_oracle PRIVATE
  "${CUTEGEN_INCLUDE_DIR}"
  "${NANOBIND_ROOT}/include"
  "${NANOBIND_ROOT}/ext/robin_map/include"
  "${PYTHON_INCLUDE_DIR}")
if(WIN32)
  target_link_libraries(_cutegen_oracle PRIVATE Python3::Module)
  target_compile_options(_cutegen_oracle PRIVATE /utf-8 /bigobj)
  set_target_properties(_cutegen_oracle PROPERTIES PREFIX "" SUFFIX ".pyd")
endif()
set_target_properties(_cutegen_oracle PROPERTIES
  OUTPUT_NAME "_cutegen_oracle"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
