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
#include <rt_tm/cpu/cpu_op_core.hpp>
#include <rt_tm/cpu/cpu_arch.hpp>

namespace rt_tm {

	template<auto... value> struct error_printer_impl;

	template<bool value, auto... value_to_test> struct static_assert_printer {
		RT_TM_FORCE_INLINE static constexpr bool impl() {
			if constexpr (!value) {
				error_printer_impl<value_to_test...>::failure_value;
				return false;
			} else {
				return true;
			}
		}
	};

	template<device_type dev_type, impl_indices indices_new, kernel_type type, typename... operand_types> struct kernel_dispatcher {
		static constexpr bool specialized{ false };
		//kernel_traits<type, operand_types...>;
		template<typename... value_types> RT_TM_FORCE_INLINE static void impl(core<value_types...>& params) {
			//kernel_dispatcher_impl<indices_new, type, value_types::output_type...>::impl();
		}
		//static_assert(specialized, "Sorry, but you need to specialize kernel_dispatcher for one of your kernel calls.");
	};

}