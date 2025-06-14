/*
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
#define RT_TM_CPU_INSTRUCTIONS 1

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
