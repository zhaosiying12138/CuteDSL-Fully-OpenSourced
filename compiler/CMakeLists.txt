# selfcute dialects: kernel / pipeline / sm120 — CuTeDSL-Fully-OpenSourced
# Built out-of-tree against the pinned LLVM/MLIR in build-llvm/ (same one
# cutlass_compiler uses), so selfcute ops interoperate with cute + base.
cmake_minimum_required(VERSION 3.20)
project(SelfCute LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
  add_compile_options(/permissive- /Zc:__cplusplus /Zc:preprocessor
                      /Zc:externConstexpr /Zc:inline)
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cMake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddMLIR)
include(AddLLVM)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
# tablegen .inc output dirs (declared by each dialect's CMakeLists)
include_directories(${PROJECT_BINARY_DIR}/Dialect/Kernel
                    ${PROJECT_BINARY_DIR}/Dialect/Pipeline
                    ${PROJECT_BINARY_DIR}/Dialect/SM120)
link_directories(${LLVM_BUILD_LIBRARY_DIR_INTDIR})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(Dialect/Kernel)
add_subdirectory(Dialect/Pipeline)
add_subdirectory(Dialect/SM120)
add_subdirectory(Registration)
add_subdirectory(tools/selfcute-opt)

# LIT tests
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test")
  add_subdirectory(test)
endif()
