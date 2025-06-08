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

message(STATUS "RT-TM: Configuring AVX2 optimizations")

file(GLOB_RECURSE RT_TM_AVX2_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(rt_tm_avx2 STATIC ${RT_TM_AVX2_SOURCES})
add_library(rt_tm::avx2 ALIAS rt_tm_avx2)

rt_tm_configure_x64_variant(rt_tm_avx2)

target_compile_options(rt_tm_avx2 PRIVATE
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-mavx2>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-mfma>"
    "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>"
)

target_compile_definitions(rt_tm_avx2 PUBLIC
    "RT_TM_AVX2"
)