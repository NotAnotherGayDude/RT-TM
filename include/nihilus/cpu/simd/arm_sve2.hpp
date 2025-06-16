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

#include <nihilus/common/kernel_traits.hpp>

#if defined(NIHILUS_SVE2)

	#include <arm_sve.h>

namespace nihilus {

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::copy, transform_type, core_type, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::copy, transform_type, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::copy, transform_type, core_type, half, float>
		: public kernel_traits<2, core_type::type, kernel_type::copy, transform_type, core_type, half, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::cont, transform_type, core_type, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::cont, transform_type, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::silu, transform_type, core_type, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::silu, transform_type, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::rms_norm, transform_type, core_type, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::rms_norm, transform_type, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_traits<2, core_type::type, kernel_type::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_traits<2, core_type::type, kernel_type::get_rows, transform_type, core_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul, transform_type, core_type, float, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::mul, transform_type, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_traits<2, core_type::type, kernel_type::mul, transform_type, core_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_traits<2, core_type::type, kernel_type::mul_mat, transform_type, core_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::mul_mat, transform_type, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_traits<2, core_type::type, kernel_type::mul_mat, transform_type, core_type, float, half, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::softmax, transform_type, core_type, float, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::softmax, transform_type, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::add, transform_type, core_type, float, float, float>
		: public kernel_traits<2, core_type::type, kernel_type::add, transform_type, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_traits<2, core_type::type, kernel_type::rope, transform_type, core_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02, const typename core_type::input_type03& input03) {
		}
	};	

};

#endif
