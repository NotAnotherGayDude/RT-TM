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

#include <rt_tm/common/common.hpp>
#include <rt_tm/common/param_api.hpp>
#include <rt_tm/common/core_traits.hpp>
#include <rt_tm/common/kernel_traits.hpp>

namespace rt_tm {

	template<core_traits_type... types> struct core;

	template<typename value_type>
	concept core_type = is_specialization_v<value_type, core>;

	template<core_traits_type output_type> struct core<output_type> : output_type {
		using traits_type = output_type;
		RT_TM_FORCE_INLINE core() noexcept {};
	};

	template<core_traits_type output_type, core_type input_type01> struct core<output_type, input_type01> : output_type {
		using traits_type		 = output_type;
		using kernel_traits_type = kernel_traits<output_type::type, output_type, input_type01>;
		RT_TM_FORCE_INLINE core() noexcept {};
		RT_TM_FORCE_INLINE core(input_type01& input01_new) : input01{ &input01_new } {};
		input_type01* input01{};
	};

	template<core_traits_type output_type, core_type input_type01, core_type input_type02> struct core<output_type, input_type01, input_type02> : public output_type {
		using traits_type		 = output_type;
		using kernel_traits_type = kernel_traits<output_type::type, output_type, input_type01, input_type02>;
		RT_TM_FORCE_INLINE core() noexcept {};
		RT_TM_FORCE_INLINE core(input_type01& input01_new, input_type02& input02_new) : input01{ &input01_new }, input02{ &input02_new } {};
		input_type01* input01{};
		input_type02* input02{};
	};

	template<core_traits_type output_type, core_type input_type01, core_type input_type02, core_type input_type03>
	struct core<output_type, input_type01, input_type02, input_type03> : public output_type {
		using traits_type		 = output_type;
		using kernel_traits_type = kernel_traits<output_type::type, output_type, input_type01, input_type02, input_type03>;
		RT_TM_FORCE_INLINE core() noexcept {};
		RT_TM_FORCE_INLINE core(input_type01& input01_new, input_type02& input02_new, input_type03& input03_new)
			: input01{ &input01_new }, input02{ &input02_new }, input03{ &input03_new } {};
		input_type01* input01{};
		input_type02* input02{};
		input_type03* input03{};
	};
}