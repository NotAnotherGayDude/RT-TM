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

#include <rt_tm/common/config.hpp>
#include <rt_tm/common/layer.hpp>
#include <unordered_set>

namespace rt_tm {

	template<model_config config, model_arch arch, typename model_traits_type, typename kernel_type_profile_traits_type> struct block {};

	template<model_config config, typename model_traits_type, typename kernel_type_profile_traits_type>
	struct block<config, model_arch::llama, model_traits_type, kernel_type_profile_traits_type> {
		using arch_traits = model_traits_type;
		using layer_type  = layer<config, model_arch::llama, model_traits_type, kernel_type_profile_traits_type>;
		layer_type layers{};

		template<typename memory_buffer_type> RT_TM_FORCE_INLINE void map_memory(memory_buffer_type& memory_buffer) {
			layers.map_memory(memory_buffer);
		}

		static constexpr size_t get_smart_total_bytes() {
			size_t total = 0;
			/*
			total += total_bytes_size<typename layer_type::token_embedding_type>::impl();
			total += total_bytes_size<typename layer_type::rope_freqs_type>::impl();
			total += total_bytes_size<typename layer_type::output_norm_type>::impl();
			total += total_bytes_size<typename layer_type::output_type>::impl();
			total += total_bytes_size<typename layer_type::input_tokens_type>::impl();
			total += total_bytes_size<typename layer_type::pos_embd_type>::impl();
			total += total_bytes_size<typename layer_type::attn_mask_type>::impl();
			total += total_bytes_size<typename layer_type::token_embd_type>::impl();
			total += total_bytes_size<typename layer_type::attn_norm_weight_type>::impl();
			total += total_bytes_size<typename layer_type::attn_q_weight_type>::impl();
			total += total_bytes_size<typename layer_type::attn_k_weight_type>::impl();
			total += total_bytes_size<typename layer_type::attn_v_weight_type>::impl();
			total += total_bytes_size<typename layer_type::attn_out_weight_type>::impl();
			total += total_bytes_size<typename layer_type::k_cache_weight_type>::impl();
			total += total_bytes_size<typename layer_type::v_cache_weight_type>::impl();
			total += total_bytes_size<typename layer_type::norm_rms_type>::impl();
			total += total_bytes_size<typename layer_type::attn_norm_mul_type>::impl();
			total += total_bytes_size<typename layer_type::attn_q_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::attn_k_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::attn_v_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::attn_q_reshape_type>::impl();
			total += total_bytes_size<typename layer_type::attn_k_reshape_type>::impl();
			total += total_bytes_size<typename layer_type::attn_v_reshape_type>::impl();
			total += total_bytes_size<typename layer_type::attn_q_rope_type>::impl();
			total += total_bytes_size<typename layer_type::attn_k_rope_type>::impl();
			total += total_bytes_size<typename layer_type::k_cache_view_type>::impl();
			total += total_bytes_size<typename layer_type::k_cache_copy_type>::impl();
			total += total_bytes_size<typename layer_type::v_cache_view_type>::impl();
			total += total_bytes_size<typename layer_type::v_cache_copy_type>::impl();
			total += total_bytes_size<typename layer_type::v_cache_reshape_type>::impl();
			total += total_bytes_size<typename layer_type::v_cache_transpose_type>::impl();
			total += total_bytes_size<typename layer_type::k_cache_permute_type>::impl();
			total += total_bytes_size<typename layer_type::v_cache_permute_type>::impl();
			total += total_bytes_size<typename layer_type::attn_q_permute_type>::impl();
			total += total_bytes_size<typename layer_type::attn_scores_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::attn_weights_softmax_type>::impl();
			total += total_bytes_size<typename layer_type::attn_context_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::attn_context_permute_type>::impl();
			total += total_bytes_size<typename layer_type::attn_context_cont_type>::impl();
			total += total_bytes_size<typename layer_type::attn_out_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_norm_weight_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_gate_weight_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_up_weight_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_down_weight_type>::impl();
			total += total_bytes_size<typename layer_type::norm_rms_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_norm_mul_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_gate_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_up_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_gate_silu_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_intermediate_mul_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_down_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::attention_residual_type>::impl();
			total += total_bytes_size<typename layer_type::ffn_residual_type>::impl();
			total += total_bytes_size<typename layer_type::norm_rms_type>::impl();
			total += total_bytes_size<typename layer_type::output_norm_mul_type>::impl();
			total += total_bytes_size<typename layer_type::output_mul_mat_type>::impl();
			total += total_bytes_size<typename layer_type::logits_softmax_type>::impl();
			total += total_bytes_size<typename layer_type::sample_multinomial_type>::impl();*/
			return total;
		}
		static constexpr size_t total_required_bytes{ get_smart_total_bytes() };
	};

	template<model_config config, typename model_traits_type, typename kernel_type_profile_traits_type, typename index_sequence> struct block_holder;

	template<model_config config, typename model_traits_type, typename kernel_type_profile_traits_type, size_t... inputs>
	struct block_holder<config, model_traits_type, kernel_type_profile_traits_type, std::index_sequence<inputs...>> {
		using tuple_block_t = tuple<block<config, config.arch, model_traits_type, kernel_type_profile_traits_type>>;
		tuple_block_t thread_blocks{};

		static constexpr size_t total_bytes{ block<config, config.arch, model_traits_type, kernel_type_profile_traits_type>::total_required_bytes };

		template<size_t current_index = 0, typename memory_buffer_type> RT_TM_FORCE_INLINE constexpr void map_memory(memory_buffer_type& memory_buffer) const {
			if constexpr (current_index < sizeof...(inputs)) {
				map_memory<current_index + 1>(memory_buffer);
			}
		}

		template<size_t current_index = 0> RT_TM_FORCE_INLINE constexpr auto& get() const {
			return rt_tm::get<current_index>(thread_blocks);
		}
	};
}
