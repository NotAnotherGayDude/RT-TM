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
#if defined(RT_TM_AVX2)
	#pragma once

	#include <rt_tm/common/common.hpp>

namespace rt_tm {

	template<impl_indices indices, kernel_type type, typename... operand_types> struct kernel_dispatcher_impl;

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::get_rows, block_q8_0<half>, int32_t, float> {
		//RT_TM_FORCE_INLINE void impl(const block_q8_0<half>*, const int32_t*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::rms_norm, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::mul, float, block_q8_0<half>, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, const block_q8_0<half>*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::mul_mat, block_q8_0<half>, float, float> {
		//RT_TM_FORCE_INLINE void impl(const block_q8_0<half>*, const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::rope, float, int32_t, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, const int32_t*, const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::copy, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::permute, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::mul_mat, float, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::softmax, float, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::add, float, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::silu, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, float*) {
		//}
	};

	template<> struct kernel_dispatcher_impl<impl_indices{ .cpu_index = 1 }, kernel_type::mul, float, float, float> {
		//RT_TM_FORCE_INLINE void impl(const float*, const float*, float*) {
		//}
	};

};

#endif