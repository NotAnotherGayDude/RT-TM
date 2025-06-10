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
#include <rt_tm/common/array.hpp>
#include <latch>

namespace rt_tm {

	enum class llama_layer_type : uint64_t {
		weights,// block weights.
		per_model,// input per_model
		token_embedding,// Invariant: [vocab_size, embedding_dim]
		attention_block,// Invariant: ALL ops share (embedding_dim, head_count) basis
		ffn_block,// Invariant: ALL ops share (embedding_dim, fftiplier) basis
		rms_norm,// Invariant: [*, embedding_dim] → [*, embedding_dim]
		residual,// Invariant: same shape in = same shape out
		lm_head,// Invariant: [embedding_dim, vocab_size]
		count
	};

	template<op_types op_type, kernel_type type, auto layer_type, typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits;

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attn_q, kernel_type::noop, llama_layer_type::weights, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::embedding_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		output_type* data{};
	};

	template<core_traits_type base_type> struct total_bytes_size<base_type> : public base_type {
		RT_TM_FORCE_INLINE static constexpr size_t impl() {
			if constexpr (array_type<decltype(base_type::data)>) {
				return base_type::total_required_bytes * decltype(base_type::data)::size_val;
			} else {
				return base_type::total_required_bytes;
			}
		}
	};

	template<typename base_type> struct memory_map : public base_type {
		using output_type = base_type::output_type;
		RT_TM_FORCE_INLINE static constexpr void impl(base_type& value, output_type* ptr) {
			if constexpr (array_type<decltype(base_type::data)>) {
				for (size_t x = 0; x < decltype(base_type::data)::size_val; ++x) {
					value.data[x] = ptr;
				}
			} else {
				value.data = ptr;
			}
		}
	};

}
