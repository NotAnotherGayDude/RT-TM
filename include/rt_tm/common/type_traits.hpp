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

#include <rt_tm/common/data_types.hpp>
#include <rt_tm/common/common.hpp>
#include <rt_tm/common/array.hpp>

namespace rt_tm {

	struct type_traits_dynamic {
		const char* type_name{};
		uint64_t block_size{};
		uint64_t type_size{};
		bool is_quantized{};
		uint64_t n_rows{};
		data_type type{};
	};

	template<data_type> struct type_traits;

	template<> struct type_traits<data_type::int_8> {
		using value_type = int8_t;
		using quant_type = int8_t;
		inline static constexpr data_type type{ data_type::int_8 };
		inline static constexpr uint64_t type_size{ sizeof(int8_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<data_type::int_32> {
		using value_type = int32_t;
		using quant_type = int32_t;
		inline static constexpr data_type type{ data_type::int_32 };
		inline static constexpr uint64_t type_size{ sizeof(int32_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<data_type::float_32> {
		using value_type = float;
		using quant_type = float;
		inline static constexpr data_type type{ data_type::float_32 };
		inline static constexpr uint64_t type_size{ sizeof(float) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<data_type::float_16> {
		using value_type = fp16_t;
		using quant_type = fp16_t;
		inline static constexpr data_type type{ data_type::float_16 };
		inline static constexpr uint64_t type_size{ sizeof(fp16_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<data_type::q8_0> {
		using value_type = block_q8_0<half>;
		using quant_type = block_q8_0<half>;
		inline static constexpr data_type type{ data_type::q8_0 };
		inline static constexpr uint64_t type_size{ sizeof(block_q8_0<half>) };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ Q_SIZE };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<data_type::count> {
		inline static constexpr data_type type{ data_type::count };
		inline static constexpr uint64_t type_size{ 0 };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ 0 };
		inline static constexpr uint64_t n_rows{ 0 };
	};

	template<data_type type> struct type_traits {};

	static constexpr bool is_it_a_type(size_t index) {
		switch (index) {
			case 0: {
				return true;
			}
			case 1: {
				return true;
			}
			case 8: {
				return true;
			}
			case 24: {
				return true;
			}
			case 26: {
				return true;
			}
			case 30: {
				return true;
			}
			case 36: {
				return true;
			}
			case 37: {
				return true;
			}
			case 38: {
				return true;
			}
			case 39: {
				return true;
			}
			default: {
				return false;
			}
		}
	}

	template<size_t index = 0> static constexpr auto get_type_traits_dynamic(array<type_traits_dynamic, static_cast<size_t>(data_type::count)> array_of_traits = {}) {
		if constexpr (index < static_cast<size_t>(data_type::count)) {
			if constexpr (is_it_a_type(index)) {
				using type_traits = type_traits<static_cast<data_type>(index)>;

				array_of_traits[index] = { .type_name = nullptr,
					.block_size						  = type_traits::block_size,
					.type_size						  = type_traits::type_size,
					.is_quantized					  = type_traits::is_quantized,
					.n_rows							  = type_traits::n_rows,
					.type							  = type_traits::type };

				switch (static_cast<data_type>(index)) {
					case data_type::int_8:
						array_of_traits[index].type_name = "i8";
						break;
					case data_type::int_16:
						array_of_traits[index].type_name = "i16";
						break;
					case data_type::int_32:
						array_of_traits[index].type_name = "i32";
						break;
					case data_type::int_64:
						array_of_traits[index].type_name = "i64";
						break;
					case data_type::float_16:
						array_of_traits[index].type_name = "f16";
						break;
					case data_type::float_32:
						array_of_traits[index].type_name = "f32";
						break;
					case data_type::float_64:
						array_of_traits[index].type_name = "f64";
						break;
					case data_type::q8_0:
						array_of_traits[index].type_name = "q8_0";
						break;
					case data_type::count:
					default:
						break;
				}
			}
			return get_type_traits_dynamic<index + 1>(array_of_traits);
		}
		return array_of_traits;
	}

	inline static constexpr auto array_of_type_traits{ get_type_traits_dynamic() };

	RT_TM_FORCE_INLINE constexpr type_traits_dynamic get_type_traits(data_type type) {
		return array_of_type_traits[static_cast<size_t>(type)];
	}

}
