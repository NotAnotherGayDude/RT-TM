#pragma once

#include <rt_tm/common/common.hpp>

#if defined(RT_TM_AVX512)

namespace rt_tm {

	template<uint64_t cpu_arch_index, kernel_type type, typename transform_type, typename... operand_types> struct kernel_dispatcher_impl;

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::get_rows, transform_type, float, block_q8_0<half>, int32_t> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const block_q8_0<half>*, const int32_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::get_rows, transform_type, float, float, int32_t> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const int32_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::rms_norm, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::transpose, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::view, transform_type, int26_t, int26_t> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, int26_t*, const int26_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::mul, transform_type, float, float, block_q8_0<half>> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const block_q8_0<half>*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, float, block_q8_0<half>, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const block_q8_0<half>*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, float, int26_t, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const int26_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::rope, transform_type, float, float, int32_t, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const int32_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::copy, transform_type, int26_t, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, int26_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::permute, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::softmax, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float* output, const float* input02, const float* input02) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::add, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::silu, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::cont, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::reshape, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<2, kernel_type::mul, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

};

#endif
