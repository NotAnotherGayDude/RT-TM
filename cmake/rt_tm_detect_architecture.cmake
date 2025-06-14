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

if (UNIX OR APPLE)
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTester.sh" "#!/bin/bash
\"${CMAKE_COMMAND}\" -S ./ -B ./Build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
\"${CMAKE_COMMAND}\" --build ./Build --config=Release")
    execute_process(
        COMMAND chmod +x "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTester.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTester.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTester.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Build/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTester.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build -DCMAKE_BUILD_TYPE=Release
\"${CMAKE_COMMAND}\" --build ./Build --config=Release")
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTester.bat"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Build/Release/feature_detector.exe")
endif()

execute_process(
    COMMAND "${FEATURE_TESTER_FILE}"
    RESULT_VARIABLE RT_TM_CPU_INSTRUCTIONS_NEW
)

set(AVX_FLAG "")

math(EXPR RT_TM_CPU_INSTRUCTIONS_NUMERIC "${RT_TM_CPU_INSTRUCTIONS_NEW}")
math(EXPR RT_TM_CPU_INSTRUCTIONS 0)

function(check_instruction_set INSTRUCTION_SET_NAME INSTRUCTION_SET_FLAG INSTRUCTION_SET_NUMERIC_VALUE)
    math(EXPR INSTRUCTION_PRESENT "( ${RT_TM_CPU_INSTRUCTIONS_NUMERIC} & ${INSTRUCTION_SET_NUMERIC_VALUE} )")
    if(INSTRUCTION_PRESENT)
        set(INSTRUCTION_SET_NAME "${INSTRUCTION_SET_NAME}" PARENT_SCOPE)
        math(EXPR RT_TM_CPU_INSTRUCTIONS "( ${RT_TM_CPU_INSTRUCTIONS} | ${INSTRUCTION_SET_NUMERIC_VALUE} )")
        set(AVX_FLAG "${AVX_FLAG};${INSTRUCTION_SET_FLAG}" PARENT_SCOPE)
    endif()
endfunction()

math(EXPR INSTRUCTION_PRESENT_SVE2 "(${RT_TM_CPU_INSTRUCTIONS_NUMERIC} & 0x8)")
math(EXPR INSTRUCTION_PRESENT_AVX512 "(${RT_TM_CPU_INSTRUCTIONS_NUMERIC} & 0x2)")
math(EXPR INSTRUCTION_PRESENT_NEON "(${RT_TM_CPU_INSTRUCTIONS_NUMERIC} & 0x4)")
math(EXPR INSTRUCTION_PRESENT_AVX2 "(${RT_TM_CPU_INSTRUCTIONS_NUMERIC} & 0x1)")

if(INSTRUCTION_PRESENT_SVE2)
    set(RT_TM_CPU_INSTRUCTIONS 8)
elseif(INSTRUCTION_PRESENT_AVX512)
    set(RT_TM_CPU_INSTRUCTIONS 2)
elseif(INSTRUCTION_PRESENT_NEON)
    set(RT_TM_CPU_INSTRUCTIONS 4) 
elseif(INSTRUCTION_PRESENT_AVX2)
    set(RT_TM_CPU_INSTRUCTIONS 1)
else()
    set(RT_TM_CPU_INSTRUCTIONS 0)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    check_instruction_set("Avx2" "/arch:AVX2" 0x1)
    check_instruction_set("Avx512" "/arch:AVX512" 0x2)
    check_instruction_set("Neon" "" 0x4)
    check_instruction_set("Sve2" "" 0x8)
else()    
    check_instruction_set("Avx2" "-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2" 0x1)
    check_instruction_set("Avx512" "-mavx512f;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2" 0x2)
    check_instruction_set("Neon" "-mfpu=neon" 0x4)
    check_instruction_set("Sve2" "-march=armv8-a+sve;-msve-vector-bits=scalable;-march=armv8-a+sve+sve2" 0x8)
endif()

set(AVX_FLAG "${AVX_FLAG}" CACHE STRING "AVX flags" FORCE)
set(RT_TM_CPU_INSTRUCTIONS "${RT_TM_CPU_INSTRUCTIONS}" CACHE STRING "CPU Instruction Sets" FORCE)

file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/Include/rt_tm/cpu/simd/rt_tm_cpu_instructions.hpp" "/*
Copyright (c) 2025 RealTimeChris (Chris M.)

This file is part of software offered under a restricted-use license to a designated Licensee,
whose identity is confirmed in writing by the Author.

License Terms (Summary):
- Exclusive, non-transferable license for internal use only.
- Redistribution, sublicensing, or public disclosure is prohibited without written consent.
- Full ownership remains with the Author.
- License may terminate if unused for [X months], if materially breached, or by mutual agreement.
- No warranty is provided, express or implied.

Full license terms are provided in the LICENSE file distributed with this software.

Signed,
RealTimeChris (Chris M.)
2025
*/
#pragma once

#undef RT_TM_CPU_INSTRUCTIONS
#define RT_TM_CPU_INSTRUCTIONS ${RT_TM_CPU_INSTRUCTIONS}

#define RT_TM_AVX2_BIT (1 << 0)
#define RT_TM_AVX512_BIT (1 << 1)
#define RT_TM_NEON_BIT (1 << 2)
#define RT_TM_SVE2_BIT (1 << 3)

#if RT_TM_CPU_INSTRUCTIONS & RT_TM_AVX2_BIT
	#define RT_TM_AVX2
static constexpr size_t cpu_arch_index{ 1 };
static constexpr size_t cpu_alignment{ 32 };
#elif RT_TM_CPU_INSTRUCTIONS & RT_TM_AVX512_BIT
	#define RT_TM_AVX512
static constexpr size_t cpu_arch_index{ 2 };
static constexpr size_t cpu_alignment{ 64 };
#elif RT_TM_CPU_INSTRUCTIONS & RT_TM_NEON_BIT
	#define RT_TM_NEON
static constexpr size_t cpu_arch_index{ 1 };
static constexpr size_t cpu_alignment{ 16 };
#elif RT_TM_CPU_INSTRUCTIONS & RT_TM_SVE2_BIT
	#define RT_TM_SVE2
static constexpr size_t cpu_arch_index{ 2 };
static constexpr size_t cpu_alignment{ 64 };
#else
static constexpr size_t cpu_arch_index{ 0 };
static constexpr size_t cpu_alignment{ 16 };
#endif
")