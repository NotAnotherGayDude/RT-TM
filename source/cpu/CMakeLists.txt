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

message(STATUS "RT-TM: Configuring CPU variant subdirectories with centralized flags")

set(RT_TM_X64_BIT_MANIPULATION_FLAGS
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-mlzcnt>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-mbmi2>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-mbmi>"
)

set(RT_TM_ARM_COMMON_FLAGS
    "$<$<AND:$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>>:-march=armv8-a>"
    "$<$<AND:$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm>>:-mfloat-abi=hard>"
)

function(rt_tm_configure_cpu_variant target_name)
    target_compile_options(${target_name} PRIVATE
        ${RT_RM_COMMON_COMPILE_FLAGS}
    )
    
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
    )

    target_link_options(
        ${target_name} PRIVATE
        "${RT_RM_COMMON_LINK_OPTIONS}"
    )
    
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

function(rt_tm_configure_x64_variant target_name)
    rt_tm_configure_cpu_variant(${target_name})
    
    target_compile_options(${target_name} PRIVATE
        ${RT_TM_X64_BIT_MANIPULATION_FLAGS}
    )
endfunction()

function(rt_tm_configure_arm_variant target_name)
    rt_tm_configure_cpu_variant(${target_name})
    
    target_compile_options(${target_name} PRIVATE
        ${RT_TM_ARM_COMMON_FLAGS}
    )
endfunction()

if(RT_TM_ARCH_X64)
    message(STATUS "RT-TM: Building ALL x64 SIMD variants")
    add_subdirectory(avx_2)
    add_subdirectory(avx_512)    
elseif(RT_TM_ARCH_ARM64)
    message(STATUS "RT-TM: Building ALL ARM64 SIMD variants")
    add_subdirectory(arm_neon)
    add_subdirectory(arm_sve)
endif()

message(STATUS "RT-TM: CPU variants configured with centralized flag management")
message(STATUS "RT-TM: Runtime CPU detection will select optimal implementation")
