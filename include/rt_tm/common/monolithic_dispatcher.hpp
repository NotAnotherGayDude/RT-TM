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

#include <rt_tm/common/kernel_traits.hpp>
#include <rt_tm/common/common.hpp>
#include <rt_tm/common/core_traits.hpp>
#include <rt_tm/cpu/cpu_arch.hpp>

namespace rt_tm {

	template<impl_indices indices_new, device_type dev_type, kernel_type type, single_input core_type> struct kernel_dispatcher
		: public kernel_traits<type, core_type, typename core_type::input_type01> {
		RT_TM_FORCE_INLINE static void impl(core_type& params) {
			kernel_dispatcher_impl<indices_new.cpu_index, type, typename core_type::transform_type, typename core_type::output_type,
				typename core_type::input_type01::output_type>::impl(params.count, params.data, get_adjacent_value<core_type, 0>::impl(params).data);
		}
	};

	template<impl_indices indices_new, device_type dev_type, kernel_type type, double_input core_type> struct kernel_dispatcher<indices_new, dev_type, type, core_type>
		: public kernel_traits<type, core_type, typename core_type::input_type01, typename core_type::input_type02> {
		RT_TM_FORCE_INLINE static void impl(core_type& params) {
			kernel_dispatcher_impl<indices_new.cpu_index, type, typename core_type::transform_type, typename core_type::output_type, typename core_type::input_type01::output_type,
				typename core_type::input_type02::output_type>::impl(params.count, params.data, get_adjacent_value<core_type, 0>::impl(params).data,
				get_adjacent_value<core_type, 1>::impl(params).data);
		}
	};

	template<impl_indices indices_new, device_type dev_type, kernel_type type, triple_input core_type> struct kernel_dispatcher<indices_new, dev_type, type, core_type>
		: public kernel_traits<type, core_type, typename core_type::input_type01, typename core_type::input_type02, typename core_type::input_type03> {
		RT_TM_FORCE_INLINE static void impl(core_type& params) {
			kernel_dispatcher_impl<indices_new.cpu_index, type, typename core_type::transform_type, typename core_type::output_type, typename core_type::input_type01::output_type,
				typename core_type::input_type02::output_type, typename core_type::input_type03::output_type>::impl(params.count, params.data,
				get_adjacent_value<core_type, 0>::impl(params).data, get_adjacent_value<core_type, 1>::impl(params).data, get_adjacent_value<core_type, 2>::impl(params).data);
		}
	};

}