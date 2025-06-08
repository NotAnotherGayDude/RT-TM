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
		per_model,// input per_model
		token_embedding,// Invariant: [vocab_size, embedding_dim]
		attention_block,// Invariant: ALL ops share (embedding_dim, head_count) basis
		ffn_block,// Invariant: ALL ops share (embedding_dim, ff_multiplier) basis
		rms_norm,// Invariant: [*, embedding_dim] → [*, embedding_dim]
		residual,// Invariant: same shape in = same shape out
		lm_head,// Invariant: [embedding_dim, vocab_size]
		count
	};

	template<op_types op_type, kernel_type type, auto layer_type, typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits;

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::per_model_token_embd_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::embedding_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::per_model_rope_freqs_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::rope_freq_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::rope_dimension_count / 2, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::per_model_output_norm_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::norm_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::per_model_output_noop, kernel_type::noop, llama_layer_type::per_model, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::token_embedding_input_tokens_noop, kernel_type::noop, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::input_token_type;
		static constexpr array<size_t, 4> dims{ { 1, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::token_embedding_pos_embd_noop, kernel_type::noop, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::position_type;
		static constexpr array<size_t, 4> dims{ { 1, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::token_embedding_attn_mask_noop, kernel_type::noop, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::attention_mask_type;
		static constexpr array<size_t, 4> dims{ { 1, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::token_embedding_inp_embd_get_rows, kernel_type::get_rows, llama_layer_type::token_embedding, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::embedding_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_norm_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::norm_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_q_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_k_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_v_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_out_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_k_cache_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_v_cache_noop, kernel_type::noop, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_norm_rms_norm, kernel_type::rms_norm, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_norm_mul, kernel_type::mul, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_q_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_k_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_v_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_q_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_k_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count_kv, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_v_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count_kv, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_q_rope, kernel_type::rope, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_k_rope, kernel_type::rope, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count_kv, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_k_cache_view, kernel_type::view, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_k_cache_copy, kernel_type::copy, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_v_cache_view, kernel_type::view, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { 1, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_v_cache_copy, kernel_type::copy, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { 1, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_v_cache_reshape, kernel_type::reshape, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_v_cache_transpose, kernel_type::transpose, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { 1, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_k_cache_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, 1, model_traits_type::head_count_kv, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_v_cache_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { 1, model_traits_type::head_dim, model_traits_type::head_count_kv, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_q_permute, kernel_type::permute, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, 1, model_traits_type::head_count, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_scores_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::attention_score_type;
		static constexpr array<size_t, 4> dims{ { 1, 1, model_traits_type::head_count, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<op_types::attention_block_attn_weights_softmax, kernel_type::softmax,
		llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::attention_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, 1, model_traits_type::head_count, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<op_types::attention_block_attn_context_mul_mat, kernel_type::mul_mat,
		llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, 1, model_traits_type::head_count, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<op_types::attention_block_attn_context_permute, kernel_type::permute,
		llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_context_cont, kernel_type::cont, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_attn_out_mul_mat, kernel_type::mul_mat, llama_layer_type::attention_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::attention_block_residual_add, kernel_type::add, llama_layer_type::residual, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_norm_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::norm_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_gate_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_up_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_down_noop, kernel_type::noop, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_norm_rms_norm, kernel_type::rms_norm, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_norm_mul, kernel_type::mul, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_gate_mul_mat, kernel_type::mul_mat, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::ffn_gate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_up_mul_mat, kernel_type::mul_mat, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::ffn_up_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_gate_silu, kernel_type::silu, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::ffn_gate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_intermediate_mul, kernel_type::mul, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_ffn_down_mul_mat, kernel_type::mul_mat, llama_layer_type::ffn_block, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::ffn_down_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::ffn_block_residual_add, kernel_type::add, llama_layer_type::residual, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::output_layer_norm_rms_norm, kernel_type::rms_norm, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::output_layer_output_norm_mul, kernel_type::mul, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::output_layer_output_mul_mat, kernel_type::mul_mat, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::logit_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::vocab_size, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::output_layer_logits_softmax, kernel_type::softmax, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::probability_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::vocab_size, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<op_types::output_layer_sample_multinomial, kernel_type::noop, llama_layer_type::lm_head, model_traits_type, kernel_type_profile_traits_type> {
		using output_type = typename kernel_type_profile_traits_type::output_token_type;
		static constexpr array<size_t, 4> dims{ { 1, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>{}.total_byte_size(dims) };
		array<output_type*, model_traits_type::block_count> data{};
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
