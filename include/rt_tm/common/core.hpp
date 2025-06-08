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

namespace rt_tm {

	template<core_traits_type... types> struct core : public types... {
		RT_TM_FORCE_INLINE core() noexcept {};
	};

	template<typename output_type> struct core<output_type> : output_type {
		using traits_type = output_type;
		RT_TM_FORCE_INLINE core() noexcept {};
	};

	template<typename output_type, typename input_type01> struct core<output_type, input_type01> : output_type {
		using traits_type = output_type;
		RT_TM_FORCE_INLINE core() noexcept {};
		RT_TM_FORCE_INLINE core(input_type01& input01_new) : input01{ &input01_new } {};
		input_type01* input01{};
	};

	template<typename output_type, typename input_type01> core(input_type01) -> core<output_type, input_type01>;

	template<typename output_type, typename input_type01, typename input_type02> struct core<output_type, input_type01, input_type02> : public output_type {
		using traits_type = output_type;
		RT_TM_FORCE_INLINE core() noexcept {};
		RT_TM_FORCE_INLINE core(input_type01& input01_new, input_type02& input02_new) : input01{ &input01_new }, input02{ &input02_new } {};
		input_type01* input01{};
		input_type02* input02{};
	};

	template<typename output_type, typename input_type01, typename input_type02> core(input_type01, input_type02) -> core<output_type, input_type01, input_type02>;

	template<typename output_type, typename input_type01, typename input_type02, typename input_type03> struct core<output_type, input_type01, input_type02, input_type03>
		: public output_type {
		using traits_type = output_type;
		RT_TM_FORCE_INLINE core() noexcept {};
		RT_TM_FORCE_INLINE core(input_type01& input01_new, input_type02& input02_new, input_type03& input03_new)
			: input01{ &input01_new }, input02{ &input02_new }, input03{ &input03_new } {};
		input_type01* input01{};
		input_type02* input02{};
		input_type03* input03{};
	};

	template<typename output_type, typename input_type01> decltype(auto) create_core(input_type01& args) {
		return core<output_type, input_type01>{ args };
	};

	template<typename output_type, typename input_type01,  typename input_type02> decltype(auto) create_core(input_type01& args01, input_type02& args02) {
		return core<output_type, input_type01, input_type02>{ args01, args02 };
	};

}