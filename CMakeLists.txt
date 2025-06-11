# Copyright (c) 2025 RealTimeChris (Chris M.)
# 
# This file is part of software offered under a restricted-use license to a designated Licensee,
# whose identity is confirmed in writing by the Author.
# 
# License Terms (Summary):
# - Exclusive, non-transferable license for internal use only.
# - Redistribution, sublicensing, or public disclosure is prohibited without written consent.
# - Full ownership remains with the Author.
# - License may terminate if unused for [X months], if materially breached, or by mutual agreement.
# - No warranty is provided, express or implied.
# 
# Full license terms are provided in the LICENSE file distributed with this software.
# 
# Signed,
# RealTimeChris (Chris M.)
# 2025
# */

cmake_minimum_required(VERSION 3.28)

set(PROJECT_NAME "rt_tm")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(ENV{CMAKE_INSTALL_PREFIX} "${CMAKE_SOURCE_DIR}/../Install")

project(
    "${PROJECT_NAME}"
    VERSION "1.0.0"
    DESCRIPTION "RT-TM: RealTime Tensor Math - You trained the model. RT-TM gives it a voice."
    LANGUAGES CXX
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(RT_TM_ARCH_X64 TRUE)
    set(RT_TM_BUILD_ALL_X64_VARIANTS TRUE)
    message(STATUS "RT-TM: Building ALL x64 SIMD variants for runtime selection")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(RT_TM_ARCH_ARM64 TRUE)
    set(RT_TM_BUILD_ALL_ARM_VARIANTS TRUE)
    message(STATUS "RT-TM: Building ALL ARM64 SIMD variants for runtime selection")
endif()

message(STATUS "RT-TM: Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "RT-TM: Universal binary strategy: Runtime CPU detection")

add_library("${PROJECT_NAME}" INTERFACE)
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

file(GLOB_RECURSE RT_TM_HEADERS CONFIGURE_DEPENDS 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rt_tm/*.hpp"
)

set_target_properties(
    "${PROJECT_NAME}" PROPERTIES
    OUTPUT_NAME "rt_tm"
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    INTERFACE_COMPILE_FEATURES cxx_std_23
)

target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

set(RT_RM_COMMON_COMPILE_FLAGS
    "$<$<BOOL:${PGO_GENERATE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-generate>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fno-trapping-math>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fno-signed-zeros>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-funroll-loops>>"
    "$<$<BOOL:${PGO_USE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-use>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-ffast-math>>"
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:${RT_TM_UBSAN_ENABLED}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>""$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wnull-dereference>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-O3>>"
    "$<$<CONFIG:DEBUG>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-O0>>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprefetch-loop-arrays>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-falign-loops=32>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-ftree-vectorize>"
    #"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${RT_TM_ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wuninitialized>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wconversion>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wpedantic>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wshadow>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wextra>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wall>"	
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/fp:fast>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/Ob2>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/GL>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/Gy>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/O2>>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wuninitialized>"
    "$<$<CXX_COMPILER_ID:MSVC>:/permissive->"
	"$<$<CXX_COMPILER_ID:GNU>:-Wconversion>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Zc:lambda>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wshadow>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wextra>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Wall>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wall>"
	"$<$<CXX_COMPILER_ID:MSVC>:/W4>"
)

set(RT_RM_COMMON_LINK_OPTIONS
    "$<$<BOOL:${PGO_GENERATE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-generate>>"
    "$<$<BOOL:${PGO_USE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-use>>"
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>,$<STREQUAL:${RT_TM_ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:${RT_TM_UBSAN_ENABLED}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<CONFIG:RELEASE>:$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-flto=thin>>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:${RT_TM_ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<CONFIG:RELEASE>:$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wl,-O3>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/OPT:ICF>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/OPT:REF>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:GNU>:-Wl,-O3>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/LTCG>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:GNU>:-flto>>"
    "$<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO>"
)

target_compile_options(
    "${PROJECT_NAME}" INTERFACE
    "${RT_RM_COMMON_COMPILE_FLAGS}"
)

target_link_options(
    "${PROJECT_NAME}" INTERFACE
    "${RT_RM_COMMON_LINK_OPTIONS}"
)

target_compile_definitions(
    "${PROJECT_NAME}" INTERFACE 
    "$<$<STREQUAL:${RT_TM_ASAN_ENABLED},TRUE>:RT_TM_ASAN_ENABLED>"
)

add_subdirectory(source/cpu)

if(RT_TM_ARCH_X64)
    target_link_libraries("${PROJECT_NAME}" INTERFACE rt_tm::avx2 rt_tm::avx512)
    message(STATUS "RT-TM: Main library linked with ALL x64 variants")
elseif(RT_TM_ARCH_ARM64)
    target_link_libraries("${PROJECT_NAME}" INTERFACE rt_tm::arm_neon rt_tm::arm_sve)
    message(STATUS "RT-TM: Main library linked with ALL ARM64 variants")
endif()

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/rt_tm/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/${CONFIG_FILE_NAME}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
    INSTALL_DESTINATION "share/rt_tm"
    PATH_VARS EXPORTED_TARGETS_FILE_PATH
)

set(VERSION_FILE_NAME "${PROJECT_NAME}ConfigVersion.cmake")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
    "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
    DESTINATION "share/rt_tm"
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.hpp"
)

set(RT_TM_ALL_TARGETS "${PROJECT_NAME}")
if(RT_TM_ARCH_X64)
    list(APPEND RT_TM_ALL_TARGETS rt_tm_avx2 rt_tm_avx512)
elseif(RT_TM_ARCH_ARM64)
    list(APPEND RT_TM_ALL_TARGETS rt_tm_arm_neon rt_tm_arm_sve)
endif()

install(
    TARGETS ${RT_TM_ALL_TARGETS}
    EXPORT "${EXPORTED_TARGETS_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)

install(
    EXPORT "${EXPORTED_TARGETS_NAME}"
    FILE "${EXPORTED_TARGETS_FILE_NAME}"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION "share/rt_tm"
)

message(STATUS "")
message(STATUS "RT-TM Configuration Summary:")
message(STATUS "==========================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Strategy: Universal Binary with Runtime CPU Detection")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(RT_TM_BUILD_ALL_X64_VARIANTS)
    message(STATUS "x64 Variants: SSE4.2, AVX2, AVX-512 (All built)")
endif()
if(RT_TM_BUILD_ALL_ARM_VARIANTS)
    message(STATUS "ARM Variants: NEON, SVE (All built)")
endif()
message(STATUS "")
message(STATUS "RT-TM: RealTime Tensor Math")
message(STATUS "Universal binaries with runtime CPU detection")
message(STATUS "")

if (RT_TM_VS_LLAMA)
    add_subdirectory("./tests/vs-llama")
endif()