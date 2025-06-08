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

#include <rt_tm/common/arch_traits.hpp>
#include <rt_tm/common/core.hpp>

namespace rt_tm {

	template<model_config config, model_arch arch, typename model_traits_type, typename kernel_type_profile_traits_type> struct layer;

	// ============================================================================
	// PER_MODEL LAYER - Global weights and tensors
	// ============================================================================
	template<model_config config, typename model_traits_type, typename kernel_type_profile_traits_type>
	struct layer<config, model_arch::llama, model_traits_type, kernel_type_profile_traits_type> {
		using token_embedding_type =
			core_traits<op_types::per_model_token_embd_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type>;
		using rope_freqs_type =
			core_traits<op_types::per_model_rope_freqs_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type>;
		using output_norm_type =
			core_traits<op_types::per_model_output_norm_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type>;
		using output_type = core_traits<op_types::per_model_output_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type>;
		using input_tokens_type =
			core_traits<op_types::token_embedding_input_tokens_noop, kernel_type::noop, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type>;
		using pos_embd_type =
			core_traits<op_types::token_embedding_pos_embd_noop, kernel_type::noop, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type>;
		using attn_mask_type =
			core_traits<op_types::token_embedding_attn_mask_noop, kernel_type::noop, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type>;
		using token_embd_type =
			core_traits<op_types::token_embedding_inp_embd_get_rows, kernel_type::get_rows, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type>;
		using attn_norm_weight_type =
			core_traits<op_types::attention_block_attn_norm_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_q_weight_type =
			core_traits<op_types::attention_block_attn_q_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_k_weight_type =
			core_traits<op_types::attention_block_attn_k_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_v_weight_type =
			core_traits<op_types::attention_block_attn_v_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_out_weight_type =
			core_traits<op_types::attention_block_attn_out_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using k_cache_weight_type =
			core_traits<op_types::attention_block_k_cache_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using v_cache_weight_type =
			core_traits<op_types::attention_block_v_cache_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_norm_rms_type =
			core_traits<op_types::attention_block_norm_rms_norm, kernel_type::rms_norm, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_norm_mul_type =
			core_traits<op_types::attention_block_attn_norm_mul, kernel_type::mul, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_q_mul_mat_type =
			core_traits<op_types::attention_block_attn_q_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_k_mul_mat_type =
			core_traits<op_types::attention_block_attn_k_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_v_mul_mat_type =
			core_traits<op_types::attention_block_attn_v_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_q_reshape_type =
			core_traits<op_types::attention_block_attn_q_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_k_reshape_type =
			core_traits<op_types::attention_block_attn_k_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_v_reshape_type =
			core_traits<op_types::attention_block_attn_v_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_q_rope_type =
			core_traits<op_types::attention_block_attn_q_rope, kernel_type::rope, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_k_rope_type =
			core_traits<op_types::attention_block_attn_k_rope, kernel_type::rope, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using k_cache_view_type =
			core_traits<op_types::attention_block_k_cache_view, kernel_type::view, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using k_cache_copy_type =
			core_traits<op_types::attention_block_k_cache_copy, kernel_type::copy, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using v_cache_view_type =
			core_traits<op_types::attention_block_v_cache_view, kernel_type::view, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using v_cache_copy_type =
			core_traits<op_types::attention_block_v_cache_copy, kernel_type::copy, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using v_cache_reshape_type =
			core_traits<op_types::attention_block_v_cache_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using v_cache_transpose_type =
			core_traits<op_types::attention_block_v_cache_transpose, kernel_type::transpose, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using k_cache_permute_type =
			core_traits<op_types::attention_block_k_cache_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using v_cache_permute_type =
			core_traits<op_types::attention_block_v_cache_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_q_permute_type =
			core_traits<op_types::attention_block_attn_q_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_scores_mul_mat_type =
			core_traits<op_types::attention_block_attn_scores_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_weights_softmax_type = core_traits<op_types::attention_block_attn_weights_softmax, kernel_type::softmax, llama_layer_type::attention_block, model_traits_type,
			kernel_type_profile_traits_type>;
		using attn_context_mul_mat_type = core_traits<op_types::attention_block_attn_context_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type,
			kernel_type_profile_traits_type>;
		using attn_context_permute_type = core_traits<op_types::attention_block_attn_context_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type,
			kernel_type_profile_traits_type>;
		using attn_context_cont_type =
			core_traits<op_types::attention_block_attn_context_cont, kernel_type::cont, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using attn_out_mul_mat_type =
			core_traits<op_types::attention_block_attn_out_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_norm_weight_type =
			core_traits<op_types::ffn_block_ffn_norm_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_gate_weight_type =
			core_traits<op_types::ffn_block_ffn_gate_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_up_weight_type = core_traits<op_types::ffn_block_ffn_up_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_down_weight_type =
			core_traits<op_types::ffn_block_ffn_down_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_norm_rms_type =
			core_traits<op_types::ffn_block_norm_rms_norm, kernel_type::rms_norm, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_norm_mul_type = core_traits<op_types::ffn_block_ffn_norm_mul, kernel_type::mul, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_gate_mul_mat_type =
			core_traits<op_types::ffn_block_ffn_gate_mul_mat, kernel_type::mul_mat, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_up_mul_mat_type =
			core_traits<op_types::ffn_block_ffn_up_mul_mat, kernel_type::mul_mat, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_gate_silu_type =
			core_traits<op_types::ffn_block_ffn_gate_silu, kernel_type::silu, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_intermediate_mul_type =
			core_traits<op_types::ffn_block_ffn_intermediate_mul, kernel_type::mul, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_down_mul_mat_type =
			core_traits<op_types::ffn_block_ffn_down_mul_mat, kernel_type::mul_mat, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type>;
		using attention_residual_type =
			core_traits<op_types::attention_block_residual_add, kernel_type::add, llama_layer_type::residual, model_traits_type, kernel_type_profile_traits_type>;
		using ffn_residual_type = core_traits<op_types::ffn_block_residual_add, kernel_type::add, llama_layer_type::residual, model_traits_type, kernel_type_profile_traits_type>;
		using norm_rms_type =
			core_traits<op_types::output_layer_norm_rms_norm, kernel_type::rms_norm, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type>;
		using output_norm_mul_type =
			core_traits<op_types::output_layer_output_norm_mul, kernel_type::mul, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type>;
		using output_mul_mat_type =
			core_traits<op_types::output_layer_output_mul_mat, kernel_type::mul_mat, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type>;
		using logits_softmax_type =
			core_traits<op_types::output_layer_logits_softmax, kernel_type::softmax, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type>;
		using sample_multinomial_type =
			core_traits<op_types::output_layer_sample_multinomial, kernel_type::noop, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type>;

		template<typename memory_buffer_type> RT_TM_FORCE_INLINE void map_memory(memory_buffer_type& memory_buffer) {
			memory_map<token_embedding_type>::impl(token_embedding, memory_buffer.claim_memory(token_embedding.total_required_bytes));
			memory_map<rope_freqs_type>::impl(rope_freqs, memory_buffer.claim_memory(rope_freqs.total_required_bytes));
		}
		core<token_embedding_type> token_embedding{};
		core<rope_freqs_type> rope_freqs{};
		core<output_norm_type> output_norm{};
		core<output_type, output_norm_type, rope_freqs_type> output{ output_norm, rope_freqs };

		core<input_tokens_type> input_tokens{};
		core<pos_embd_type> pos_embd{};
		core<attn_mask_type> attn_mask{};
		core<token_embd_type> token_embd{};
		// Weight matrices

		// Pre-attention normalization

		// QKV projections

		// Multi-head reshaping

		// Rotary position embedding

		// KV cache operations

		// Attention computation


		// Attention output projection

		// Weight matrix instances
		core<attn_norm_weight_type> attn_norm_weight{};
		core<attn_q_weight_type> attn_q_weight{};
		core<attn_k_weight_type> attn_k_weight{};
		core<attn_v_weight_type> attn_v_weight{};
		core<attn_out_weight_type> attn_out_weight{};
		core<k_cache_weight_type> k_cache{};
		core<v_cache_weight_type> v_cache{};

		// Operation instances
		core<attn_norm_rms_type> attn_norm_rms{};
		core<attn_norm_mul_type> attn_norm_mul{};
		core<attn_q_mul_mat_type> attn_q_mul_mat{};
		core<attn_k_mul_mat_type> attn_k_mul_mat{};
		core<attn_v_mul_mat_type> attn_v_mul_mat{};
		core<attn_q_reshape_type> attn_q_reshape{};
		core<attn_k_reshape_type> attn_k_reshape{};
		core<attn_v_reshape_type> attn_v_reshape{};
		core<attn_q_rope_type> attn_q_rope{};
		core<attn_k_rope_type> attn_k_rope{};
		core<k_cache_view_type> k_cache_view{};
		core<k_cache_copy_type> k_cache_copy{};
		core<v_cache_view_type> v_cache_view{};
		core<v_cache_copy_type> v_cache_copy{};
		core<v_cache_reshape_type> v_cache_reshape{};
		core<v_cache_transpose_type> v_cache_transpose{};
		core<k_cache_permute_type> k_cache_permute{};
		core<v_cache_permute_type> v_cache_permute{};
		core<attn_q_permute_type> attn_q_permute{};
		core<attn_scores_mul_mat_type> attn_scores_mul_mat{};
		core<attn_weights_softmax_type> attn_weights_softmax{};
		core<attn_context_mul_mat_type> attn_context_mul_mat{};
		core<attn_context_permute_type> attn_context_permute{};
		core<attn_context_cont_type> attn_context_cont{};
		core<attn_out_mul_mat_type> attn_out_mul_mat{};
		// Weight matrices

		// Pre-FFN normalization

		// SwiGLU operations

		// Weight matrix instances
		core<ffn_norm_weight_type> ffn_norm_weight{};
		core<ffn_gate_weight_type> ffn_gate_weight{};
		core<ffn_up_weight_type> ffn_up_weight{};
		core<ffn_down_weight_type> ffn_down_weight{};

		// Operation instances
		core<ffn_norm_rms_type> ffn_norm_rms{};
		core<ffn_norm_mul_type> ffn_norm_mul{};
		core<ffn_gate_mul_mat_type> ffn_gate_mul_mat{};
		core<ffn_up_mul_mat_type> ffn_up_mul_mat{};
		core<ffn_gate_silu_type> ffn_gate_silu{};
		core<ffn_intermediate_mul_type> ffn_intermediate_mul{};
		core<ffn_down_mul_mat_type> ffn_down_mul_mat{};

		core<attention_residual_type> attention_residual{};
		core<ffn_residual_type> ffn_residual{};

		// NOOP tensors (sampling parameters)
		static constexpr size_t total_per_model_bytes{ input_tokens_type::total_required_bytes + pos_embd_type::total_required_bytes + attn_mask_type::total_required_bytes +
			token_embedding_type::total_required_bytes + rope_freqs_type::total_required_bytes + output_norm_type::total_required_bytes + output_type::total_required_bytes +
			attn_norm_weight_type::total_required_bytes + attn_q_weight_type::total_required_bytes + attn_k_weight_type::total_required_bytes +
			attn_v_weight_type::total_required_bytes + attn_out_weight_type::total_required_bytes + k_cache_weight_type::total_required_bytes +
			v_cache_weight_type::total_required_bytes + ffn_norm_weight_type::total_required_bytes + ffn_gate_weight_type::total_required_bytes +
			ffn_up_weight_type::total_required_bytes + ffn_down_weight_type::total_required_bytes + sample_multinomial_type::total_required_bytes };

		// Computational operations
		static constexpr size_t total_per_layer_bytes{ token_embd_type::total_required_bytes + attn_norm_rms_type::total_required_bytes + attn_norm_mul_type::total_required_bytes +
			attn_q_mul_mat_type::total_required_bytes + attn_k_mul_mat_type::total_required_bytes + attn_v_mul_mat_type::total_required_bytes +
			attn_q_reshape_type::total_required_bytes + attn_k_reshape_type::total_required_bytes + attn_v_reshape_type::total_required_bytes +
			attn_q_rope_type::total_required_bytes + attn_k_rope_type::total_required_bytes + k_cache_view_type::total_required_bytes + k_cache_copy_type::total_required_bytes +
			v_cache_view_type::total_required_bytes + v_cache_copy_type::total_required_bytes + v_cache_reshape_type::total_required_bytes +
			v_cache_transpose_type::total_required_bytes + k_cache_permute_type::total_required_bytes + v_cache_permute_type::total_required_bytes +
			attn_q_permute_type::total_required_bytes + attn_scores_mul_mat_type::total_required_bytes + attn_weights_softmax_type::total_required_bytes +
			attn_context_mul_mat_type::total_required_bytes + attn_context_permute_type::total_required_bytes + attn_context_cont_type::total_required_bytes +
			attn_out_mul_mat_type::total_required_bytes + ffn_norm_rms_type::total_required_bytes + ffn_norm_mul_type::total_required_bytes +
			ffn_gate_mul_mat_type::total_required_bytes + ffn_up_mul_mat_type::total_required_bytes + ffn_gate_silu_type::total_required_bytes +
			ffn_intermediate_mul_type::total_required_bytes + ffn_down_mul_mat_type::total_required_bytes + attention_residual_type::total_required_bytes +
			ffn_residual_type::total_required_bytes + norm_rms_type::total_required_bytes + output_norm_mul_type::total_required_bytes + output_mul_mat_type::total_required_bytes +
			logits_softmax_type::total_required_bytes };

		core<norm_rms_type> norm_rms{};
		core<output_norm_mul_type> output_norm_mul{};
		core<output_mul_mat_type> output_mul_mat{};
		core<logits_softmax_type> logits_softmax{};
		core<sample_multinomial_type> sample_multinomial{};
	};
}
