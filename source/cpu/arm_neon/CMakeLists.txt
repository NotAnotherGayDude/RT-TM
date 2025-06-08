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

message(STATUS "RT-TM: Configuring ARM NEON optimizations")

file(GLOB_RECURSE RT_TM_ARM_NEON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(rt_tm_arm_neon STATIC ${RT_TM_ARM_NEON_SOURCES})
add_library(rt_tm::arm_neon ALIAS rt_tm_arm_neon)

rt_tm_configure_arm_variant(rt_tm_arm_neon)

target_compile_options(rt_tm_arm_neon PRIVATE
    "$<$<AND:$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm>>:-mfpu=neon>"
)

target_compile_definitions(rt_tm_arm_neon PUBLIC
    "RT_TM_ARM_NEON"
)