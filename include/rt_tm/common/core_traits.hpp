/*
Copyright (c) 2025 RealTimeChris (Chris model_traits_type.)

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
RealTimeChris (Chris model_traits_type.)
2025
*/

#pragma once

#include <rt_tm/common/kernel_traits.hpp>
#include <rt_tm/common/kernel_type_profile_traits.hpp>
#include <rt_tm/common/model_traits.hpp>
#include <rt_tm/common/common.hpp>
#include <rt_tm/common/array.hpp>
#include <latch>

namespace rt_tm {

	enum class alloc_type : uint8_t {
		single_alloc,
		per_block_alloc,
	};

	enum class layer_op_type : uint8_t {
		none,
		global_input,
		global_output,
		per_block,
	};

	template<typename type01, typename type02> struct requires_dequant_or_quant {
		static constexpr bool required{ !std::is_same_v<type01, type02> };
	};

	template<impl_indices, llama_op_types op_type, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type> struct core_traits;

	template<kernel_type kernel_type01, kernel_type kernel_type02> struct output_transform {};

	template<> struct output_transform<kernel_type::silu, kernel_type::mul_mat> {
		template<typename value_type> RT_TM_FORCE_INLINE static void impl(value_type*, uint64_t) {};
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::token_embd_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::token_embd_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::inp_tokens, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::input_token_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_tokens };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::inp_pos, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::position_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_pos };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::inp_out_ids, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::output_token_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_out_ids };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::rope_freqs_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::rope_freqs_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::rope_dimension_count / 2, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::rope_freqs_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::output_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::output_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::output_norm_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::output_norm_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::output_norm_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_q_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::query_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_q_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_k_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::key_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, (model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_k_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_v_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::value_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, (model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_v_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_output_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::attn_output_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_output_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_norm_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::attn_norm_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_norm_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_gate_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_gate_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_up_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_up_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_up_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_down_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_down_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_down_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_norm_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_norm_weight_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_norm_weight };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::cache_k, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::cache_k };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::cache_v, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::cache_v };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kq_mask, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using output_type													   = typename kernel_type_profile_traits_type::kq_mask_type;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::kq_mask };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::inp_embd, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::inp_embd, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::token_embd_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::inp_tokens, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::embedding_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_input };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::inp_embd };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::inp_embd, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::norm };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type		 = core_traits<indices, llama_op_types::attn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01	 = core_traits<indices, llama_op_types::norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02	 = core_traits<indices, llama_op_types::attn_norm_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type	 = typename kernel_type_profile_traits_type::norm_output_type;
		using transform_type = output_transform<input_type01::krn_type, input_type02::krn_type>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::attn_norm };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::qcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::qcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::attn_q_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::attn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_count * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::qcur };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::qcur_reshaped, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::qcur_reshaped, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::qcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::reshape };
		static constexpr llama_op_types type{ llama_op_types::qcur_reshaped };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::qcur_rope, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::qcur_rope, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::qcur_reshaped, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::inp_pos, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type03 = core_traits<indices, llama_op_types::rope_freqs_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rope };
		static constexpr llama_op_types type{ llama_op_types::qcur_rope };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::attn_k_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::attn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kcur };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kcur_reshaped, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kcur_reshaped, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::reshape };
		static constexpr llama_op_types type{ llama_op_types::kcur_reshaped };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kcur_rope, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kcur_rope, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kcur_reshaped, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::inp_pos, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type03 = core_traits<indices, llama_op_types::rope_freqs_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr uint64_t depth{ std::max(std::max(input_type01::depth, input_type02::depth), input_type03::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rope };
		static constexpr llama_op_types type{ llama_op_types::kcur_rope };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::vcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::vcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::attn_v_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::attn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { (model_traits_type::head_dim * model_traits_type::head_count_kv), model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::vcur };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::k_cache_view, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::k_cache_view, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::cache_k, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::k_cache_view };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::k_cache_view_copy, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::k_cache_view_copy, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kcur_rope, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename core_traits<indices, llama_op_types::k_cache_view, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>::output_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::copy };
		static constexpr llama_op_types type{ llama_op_types::k_cache_view_copy };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::vcur_transposed, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::vcur_transposed, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::vcur, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, (model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(type_traits<output_type>::total_byte_size(dims), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::transpose };
		static constexpr llama_op_types type{ llama_op_types::vcur_transposed };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::v_cache_view, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::v_cache_view, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::cache_v, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, (model_traits_type::head_count_kv * model_traits_type::head_dim), 1, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::v_cache_view };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::v_cache_view_copy, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::v_cache_view_copy, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::vcur_transposed, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename core_traits<indices, llama_op_types::v_cache_view, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>::output_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, (model_traits_type::head_count_kv * model_traits_type::head_dim), 1, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::copy };
		static constexpr llama_op_types type{ llama_op_types::v_cache_view_copy };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::v, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::v, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::cache_v, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::scale_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, model_traits_type::head_dim, model_traits_type::head_count_kv, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::v };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::k, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::k, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::cache_k, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::scale_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::k };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::q, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::q, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::qcur_rope, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::permute };
		static constexpr llama_op_types type{ llama_op_types::q };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kq, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kq, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::k, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::q, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::attention_score_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, 1, model_traits_type::head_count, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kq };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kq_soft_max, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kq_soft_max, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kq, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::kq_mask, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::softmax_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::max_sequence_length, 1, model_traits_type::head_count, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::softmax };
		static constexpr llama_op_types type{ llama_op_types::kq_soft_max };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kqv, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kqv, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::v, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::kq_soft_max, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, 1, model_traits_type::head_count, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kqv };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kqv_merged, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kqv_merged, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kqv, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::permute };
		static constexpr llama_op_types type{ llama_op_types::kqv_merged };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kqv_merged_cont, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kqv_merged_cont, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kqv_merged, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::cont };
		static constexpr llama_op_types type{ llama_op_types::kqv_merged_cont };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::kqv_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::kqv_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::attn_output_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::kqv_merged_cont, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kqv_out };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_inp, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::ffn_inp, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kqv_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::l_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add };
		static constexpr llama_op_types type{ llama_op_types::ffn_inp };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::norm_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::norm_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::ffn_inp, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::norm_out };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type		 = core_traits<indices, llama_op_types::ffn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01	 = core_traits<indices, llama_op_types::norm_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02	 = core_traits<indices, llama_op_types::ffn_norm_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type	 = typename kernel_type_profile_traits_type::residual_type;
		using transform_type = output_transform<input_type01::krn_type, input_type02::krn_type>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_norm };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_gate, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::ffn_gate, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::ffn_gate_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::ffn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_silu, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::ffn_silu, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::ffn_gate, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::silu };
		static constexpr llama_op_types type{ llama_op_types::ffn_silu };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_up, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::ffn_up, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::ffn_up_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::ffn_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_up };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_gate_par, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type		 = core_traits<indices, llama_op_types::ffn_gate_par, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01	 = core_traits<indices, llama_op_types::ffn_silu, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02	 = core_traits<indices, llama_op_types::ffn_up, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type	 = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		using transform_type = output_transform<input_type01::krn_type, input_type02::krn_type>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate_par };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::ffn_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::ffn_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::ffn_down_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::ffn_gate_par, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_out };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::l_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::l_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::ffn_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::ffn_inp, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add };
		static constexpr llama_op_types type{ llama_op_types::l_out };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::attn_residual, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::attn_residual, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::kqv_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::inp_out_ids, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::attn_residual };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::prev_residual, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::prev_residual, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::l_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::inp_out_ids, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::prev_residual };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::final_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::final_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::l_out, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::final_norm };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::result_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type		 = core_traits<indices, llama_op_types::result_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01	 = core_traits<indices, llama_op_types::final_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02	 = core_traits<indices, llama_op_types::output_norm_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type	 = typename kernel_type_profile_traits_type::norm_output_type;
		using transform_type = output_transform<input_type01::krn_type, input_type02::krn_type>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::result_norm };
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<impl_indices indices, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	struct core_traits<indices, llama_op_types::result_output, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using transform_type												   = int32_t;
		using derived_type													   = derived_type_new;
		using model_traits_type												   = model_traits_type_new;
		using this_type	   = core_traits<indices, llama_op_types::result_output, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<indices, llama_op_types::output_weight, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<indices, llama_op_types::result_norm, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::logit_type;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> dims{ { model_traits_type::vocab_size, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr uint64_t total_required_bytes{ roundUpToMultiple(
			type_traits<output_type>::total_byte_size(dims) + (dequantization ? type_traits<output_type>::total_byte_size(dims) : 0), 64ull) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::result_output };
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_start{};
		array<latch_wrapper_holder, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		uint64_t count{ total_required_bytes / sizeof(output_type) };
	};

	template<typename derived_type_new, uint64_t index> struct get_adjacent_value {
		using derived_type		   = derived_type_new;
		using derived_derived_type = typename derived_type::derived_type;
		RT_TM_FORCE_INLINE static auto& impl(derived_type_new& core) {
			if constexpr (index == 0) {
				using input_type01 = typename derived_type::input_type01;
				return *static_cast<input_type01*>(static_cast<derived_derived_type*>(&core));
			} else if constexpr (index == 1) {
				using input_type02 = typename derived_type::input_type02;
				return *static_cast<input_type02*>(static_cast<derived_derived_type*>(&core));
			} else if constexpr (index == 2) {
				using input_type03 = typename derived_type::input_type03;
				return *static_cast<input_type03*>(static_cast<derived_derived_type*>(&core));
			}
		}
	};

	template<typename... bases> struct core_bases : bases... {
		RT_TM_FORCE_INLINE core_bases() noexcept					= default;
		RT_TM_FORCE_INLINE core_bases& operator=(core_bases&&)		= delete;
		RT_TM_FORCE_INLINE core_bases(core_bases&&)					= delete;
		RT_TM_FORCE_INLINE core_bases& operator=(const core_bases&) = delete;
		RT_TM_FORCE_INLINE core_bases(const core_bases&)			= delete;

		template<template<impl_indices, typename> typename mixin_type, impl_indices indices, typename op_entity_type, typename... arg_types>
		RT_TM_FORCE_INLINE void impl_internal(arg_types&&... args) {
			return mixin_type<indices, op_entity_type>::impl(*static_cast<op_entity_type*>(this), std::forward<arg_types>(args)...);
		}

		template<template<impl_indices, typename> typename mixin_type, impl_indices indices, typename... arg_types> RT_TM_FORCE_INLINE void impl(arg_types&&... args) {
			(impl_internal<mixin_type, indices, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<impl_indices, typename> typename mixin_type, impl_indices indices, typename op_entity_type, typename... arg_types>
		RT_TM_FORCE_INLINE static constexpr void impl_internal_constexpr(arg_types&&... args) {
			return mixin_type<indices, op_entity_type>::impl(std::forward<arg_types>(args)...);
		}

		template<template<impl_indices, typename> typename mixin_type, impl_indices indices, typename... arg_types>
		RT_TM_FORCE_INLINE static constexpr void impl_constexpr(arg_types&&... args) {
			(impl_internal_constexpr<mixin_type, indices, bases>(args...), ...);
		}
	};

	template<impl_indices indices, typename op_type_type, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type,
		typename index_sequence>
	struct get_core_traits_base;

	template<impl_indices indices, typename op_type_type, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type, uint64_t... index>
	struct get_core_traits_base<indices, op_type_type, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type, std::index_sequence<index...>> {
		using type = core_bases<core_traits<indices, static_cast<op_type_type>(index), derived_type_new, model_traits_type_new, kernel_type_profile_traits_type>...>;
	};

	template<impl_indices indices, typename op_type_type, typename derived_type_new, typename model_traits_type_new, typename kernel_type_profile_traits_type>
	using get_core_traits_base_t = typename get_core_traits_base<indices, op_type_type, derived_type_new, model_traits_type_new, kernel_type_profile_traits_type,
		std::make_index_sequence<static_cast<uint64_t>(op_type_type::count)>>::type;

}
