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

message(STATUS "Fap-Tensor: Configuring ARM SVE optimizations")

file(GLOB_RECURSE RT_TM_ARM_SVE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(rt_tm_arm_sve STATIC ${RT_TM_ARM_SVE_SOURCES})
add_library(rt_tm::arm_sve ALIAS rt_tm_arm_sve)

rt_tm_configure_arm_variant(rt_tm_arm_sve)

target_compile_options(rt_tm_arm_sve PRIVATE
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-msve-vector-bits=scalable>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-march=armv8-a+sve+sve2>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-march=armv8-a+sve>"
)
