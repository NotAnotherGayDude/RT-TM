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

#include <type_traits>
#include <concepts>

namespace rt_tm {

	template<typename value_type>
	concept uint_type = std::is_unsigned_v<std::remove_cvref_t<value_type>> && std::is_integral_v<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept int_type = std::is_signed_v<std::remove_cvref_t<value_type>> && std::is_integral_v<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept int8_type = int_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 1;

	template<typename value_type>
	concept int16_type = int_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 2;

	template<typename value_type>
	concept int32_type = int_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 4;

	template<typename value_type>
	concept int64_type = int_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 8;

	template<typename value_type>
	concept uint8_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 1;

	template<typename value_type>
	concept uint16_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 2;

	template<typename value_type>
	concept uint32_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 4;

	template<typename value_type>
	concept uint64_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(value_type) == 8;

	template<typename value_type>
	concept float_type = std::floating_point<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept float32_type = float_type<value_type> && sizeof(value_type) == 4;

	template<typename value_type>
	concept float64_type = float_type<value_type> && sizeof(value_type) == 8;

	template<typename value_type>
	concept has_size = requires(value_type value) {
		{ value.size() } -> std::same_as<typename value_type::size_type>;
	};

	template<typename value_type>
	concept has_data = requires(value_type value) {
		{ value.data() } -> std::same_as<typename value_type::pointer>;
	};

	template<typename value_type>
	concept vector_subscriptable = requires(value_type value) {
		{ value[std::declval<typename value_type::size_type>()] } -> std::same_as<typename value_type::reference>;
	};

	template<typename value_type>
	concept array_type = vector_subscriptable<value_type> && has_data<value_type> && has_size<value_type>;
		
	template<typename value_type>
	concept core_traits_type = requires(value_type) {
		typename value_type::output_type;
		value_type::data;
		value_type::dims;
	};

}
