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

	template<typename data_type> struct type_traits;

	template<typename derived_type> struct total_bytes_size {
		RT_TM_FORCE_INLINE constexpr static size_t total_byte_size(const array<size_t, 4>& dims) {
			size_t total_elements = dims[0] * dims[1] * dims[2] * dims[3];
			size_t num_blocks	  = (total_elements + derived_type::block_size - 1) / derived_type::block_size;
			return num_blocks * derived_type::type_size;
		}
	};

	template<> struct type_traits<int8_t> : public total_bytes_size<type_traits<int8_t>> {
		using value_type = int8_t;
		using quant_type = int8_t;
		inline static constexpr data_type type{ data_type::i8 };
		inline static constexpr uint64_t type_size{ sizeof(int8_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<int32_t> : public total_bytes_size<type_traits<int32_t>> {
		using value_type = int32_t;
		using quant_type = int32_t;
		inline static constexpr data_type type{ data_type::i32 };
		inline static constexpr uint64_t type_size{ sizeof(int32_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<float> : public total_bytes_size<type_traits<float>> {
		using value_type = float;
		using quant_type = float;
		inline static constexpr data_type type{ data_type::f32 };
		inline static constexpr uint64_t type_size{ sizeof(float) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<int16_t> : public total_bytes_size<type_traits<int16_t>> {
		using value_type = fp16_t;
		using quant_type = fp16_t;
		inline static constexpr data_type type{ data_type::f16 };
		inline static constexpr uint64_t type_size{ sizeof(fp16_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<block_q8_0<half>> : public total_bytes_size<type_traits<block_q8_0<half>>> {
		using value_type = block_q8_0<half>;
		using quant_type = block_q8_0<half>;
		inline static constexpr data_type type{ data_type::q8_0 };
		inline static constexpr uint64_t type_size{ sizeof(block_q8_0<half>) };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ Q_SIZE };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<void> : public total_bytes_size<type_traits<void>> {
		inline static constexpr data_type type{ data_type::count };
		inline static constexpr uint64_t type_size{ 0 };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ 0 };
		inline static constexpr uint64_t n_rows{ 0 };
	};

}
