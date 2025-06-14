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

#include <source_location>
#include <cstdint>
#include <utility>
#include <atomic>

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define RT_TM_PLATFORM_WINDOWS 1
#elif defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__)) || defined(TARGET_OS_MAC)
	#include <mach/mach.h>
	#define RT_TM_PLATFORM_MAC 1
#elif defined(__ANDROID__)
	#define RT_TM_PLATFORM_ANDROID 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define RT_TM_PLATFORM_LINUX 1
#else
	#error "Unsupported platform"
#endif

#if defined(_MSC_VER)
	#define RT_TM_COMPILER_MSVC 1
#elif defined(__clang__) || defined(__llvm__)
	#define RT_TM_COMPILER_CLANG 1
#elif defined(__GNUC__) && !defined(__clang__)
	#define RT_TM_COMPILER_GNUCXX 1
#else
	#error "Unsupported compiler"
#endif

#if defined(NDEBUG)
	#if defined(RT_TM_COMPILER_MSVC)
		#define RT_TM_INLINE inline
		#define RT_TM_FORCE_INLINE [[msvc::forceinline]] inline
	#elif defined(RT_TM_COMPILER_CLANG)
		#define RT_TM_INLINE inline
		#define RT_TM_FORCE_INLINE inline __attribute__((always_inline))
	#elif defined(RT_TM_COMPILER_GNUCXX)
		#define RT_TM_INLINE inline
		#define RT_TM_FORCE_INLINE inline __attribute__((always_inline))
	#endif
#else
	#if defined(RT_TM_COMPILER_MSVC)
		#define RT_TM_INLINE
		#define RT_TM_FORCE_INLINE
	#elif defined(RT_TM_COMPILER_CLANG)
		#define RT_TM_INLINE
		#define RT_TM_FORCE_INLINE
	#elif defined(RT_TM_COMPILER_GNUCXX)
		#define RT_TM_INLINE
		#define RT_TM_FORCE_INLINE
	#endif
#endif

#if !defined(RT_TM_LIKELY)
	#define RT_TM_LIKELY(...) (__VA_ARGS__) [[likely]]
#endif

#if !defined(RT_TM_UNLIKELY)
	#define RT_TM_UNLIKELY(...) (__VA_ARGS__) [[unlikely]]
#endif

#if !defined(RT_TM_ELSE_UNLIKELY)
	#define RT_TM_ELSE_UNLIKELY(...) __VA_ARGS__ [[unlikely]]
#endif

#if !defined(RT_TM_ALIGN)
	#define RT_TM_ALIGN(N) alignas(N)
#endif

#if defined(__x86_64__) || defined(_M_X64)
	#define RT_TM_ARCH_X86_64 1
	#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
	#define RT_TM_ARCH_ARM64 1
#else
	#error "Unsupported architecture"
#endif

RT_TM_FORCE_INLINE void rt_tm_pause() noexcept {
#if defined(RT_TM_ARCH_X86_64)
	_mm_pause();
#elif defined(RT_TM_ARCH_ARM64)
	__asm__ __volatile__("yield" ::: "memory");
#else
	__asm__ __volatile__("" ::: "memory");
#endif
}

#ifndef NDEBUG
	#define RT_TM_ASSERT(x) \
		if (!(x)) \
		rt_tm::internal_abort(#x, std::source_location::current())
#else
	#define RT_TM_ASSERT(x)
#endif
