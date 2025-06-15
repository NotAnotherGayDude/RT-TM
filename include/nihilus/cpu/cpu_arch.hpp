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

#include <nihilus/cpu/simd/avx_2.hpp>
#include <nihilus/cpu/simd/avx_512.hpp>

#include <nihilus/cpu/simd/arm_neon.hpp>
#include <nihilus/cpu/simd/arm_sve2.hpp>

namespace nihilus {

	template<uint64_t cpu_arch_index, kernel_type type, typename transform_type, typename... operand_types> struct kernel_dispatcher_impl;

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::get_rows, transform_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const block_q8_0<half>*, const int32_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::get_rows, transform_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const int32_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::rms_norm, transform_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::transpose, transform_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::view, transform_type, int16_t, int16_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, int16_t*, const int16_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::mul, transform_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const block_q8_0<half>*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::mul_mat, transform_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const block_q8_0<half>*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::mul_mat, transform_type, float, int16_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const int16_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::rope, transform_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const int32_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::copy, transform_type, int16_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, int16_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::permute, transform_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::mul_mat, transform_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::softmax, transform_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::add, transform_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::silu, transform_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::cont, transform_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::reshape, transform_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<0, kernel_type::mul, transform_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};
}