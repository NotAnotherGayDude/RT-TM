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

	enum class llama_layer_type : uint8_t {
		global_input	= 0,
		global_weights	= 1,
		global_output	= 2,
		global_cache	= 3,
		layer_weights	= 4,
		layer_norm		= 5,
		layer_attention = 6,
		layer_ffn		= 7,
		layer_residual	= 8,
		layer_cache		= 9,
	};

	template<typename value_type>
	concept blocking = requires(std::remove_cvref_t<value_type> value) {
		value_type::sync_flag_end;
		value_type::sync_flag_start;
	};

	template<typename value_type>
	concept no_input = requires(value_type) { typename std::remove_cvref_t<value_type>::output_type; };

	template<typename value_type>
	concept single_input = requires(value_type) {
		typename std::remove_cvref_t<value_type>::output_type;
		typename std::remove_cvref_t<value_type>::input_type01;
	};

	template<typename value_type>
	concept double_input = requires(value_type) {
		typename std::remove_cvref_t<value_type>::output_type;
		typename std::remove_cvref_t<value_type>::input_type01;
		typename std::remove_cvref_t<value_type>::input_type02;
	} && !single_input<value_type>;

	template<typename value_type>
	concept triple_input = requires(value_type) {
		typename std::remove_cvref_t<value_type>::output_type;
		typename std::remove_cvref_t<value_type>::input_type01;
		typename std::remove_cvref_t<value_type>::input_type02;
		typename std::remove_cvref_t<value_type>::input_type03;
	} && !double_input<value_type>;

	template<typename value_type>
	concept single_input_blocking = single_input<value_type> && blocking<value_type>;

	template<typename value_type>
	concept double_input_blocking = double_input<value_type> && blocking<value_type>;

	template<typename value_type>
	concept triple_input_blocking = triple_input<value_type> && blocking<value_type>;

	template<typename value_type>
	concept active_thread = single_input<value_type> || double_input<value_type> || triple_input<value_type> || single_input_blocking<value_type> ||
		double_input_blocking<value_type> || triple_input_blocking<value_type>;

	template<single_input base_type> struct cpu_core_thread {
		RT_TM_FORCE_INLINE cpu_core_thread(base_type* core_new) noexcept : core{ core_new } {};
		using output_type = base_type::output_type;
		size_t thread_count{};
		size_t thread_index{};
		base_type* core{};
		RT_TM_FORCE_INLINE void impl() {
		}
	};

	template<single_input_blocking base_type> struct cpu_core_thread<base_type> {
		RT_TM_FORCE_INLINE cpu_core_thread(base_type* core_new) noexcept : core{ core_new } {};
		using output_type = base_type::output_type;
		size_t thread_count{};
		size_t thread_index{};
		base_type* core{};
		RT_TM_FORCE_INLINE void impl() {
		}
	};

	template<double_input base_type> struct cpu_core_thread<base_type> {
		RT_TM_FORCE_INLINE cpu_core_thread(base_type* core_new) noexcept : core{ core_new } {};
		using output_type = base_type::output_type;
		size_t thread_count{};
		size_t thread_index{};
		base_type* core{};
		RT_TM_FORCE_INLINE void impl() {
		}
	};

	template<double_input_blocking base_type> struct cpu_core_thread<base_type> {
		RT_TM_FORCE_INLINE cpu_core_thread(base_type* core_new) noexcept : core{ core_new } {};
		using output_type = base_type::output_type;
		size_t thread_count{};
		size_t thread_index{};
		base_type* core{};
		RT_TM_FORCE_INLINE void impl() {
		}
	};

	template<triple_input base_type> struct cpu_core_thread<base_type> {
		RT_TM_FORCE_INLINE cpu_core_thread(base_type* core_new) noexcept : core{ core_new } {};
		using output_type = base_type::output_type;
		size_t thread_count{};
		size_t thread_index{};
		base_type* core{};
		RT_TM_FORCE_INLINE void impl() {
		}
	};

	template<triple_input_blocking base_type> struct cpu_core_thread<base_type> {
		RT_TM_FORCE_INLINE cpu_core_thread(base_type* core_new) noexcept : core{ core_new } {};
		using output_type = base_type::output_type;
		base_type* core{};
		RT_TM_FORCE_INLINE void impl() {
		}
	};

	template<llama_op_types op_type, typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits;

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::inp_embd, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using this_type														   = core_traits<llama_op_types::inp_embd, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01													   = core_traits<llama_op_types::token_embd_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02													   = core_traits<llama_op_types::inp_tokens, model_traits_type, kernel_type_profile_traits_type>;
		using output_type													   = typename kernel_type_profile_traits_type::embedding_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::get_rows };
		std::vector<cpu_core_thread<this_type>> threads{};
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::token_embd_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::embedding_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::vocab_size, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::inp_tokens, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::input_token_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::inp_pos, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::position_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::inp_out_ids, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::output_token_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::rope_freqs_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::rope_freq_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::rope_dimension_count / 2, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::output_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::output_norm_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::norm_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_q_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::head_count* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_k_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_v_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_output_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::attention_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count * model_traits_type::head_dim, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_norm_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::norm_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_gate_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_gate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_up_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_up_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_down_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::ffn_down_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::embedding_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_norm_weight, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::norm_weight_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, 1, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::cache_k, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::cache_v, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::kq_mask, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;
		using output_type													   = typename kernel_type_profile_traits_type::attention_mask_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::none };
		output_type* data{};
	};

	// NORM OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::norm, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::inp_embd, model_traits_type, kernel_type_profile_traits_type>;// or ffn_inp depending on layer
		using output_type  = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::rms_norm };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_norm, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::attn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::attn_norm_weight, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	// QUERY OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::qcur, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::qcur, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::attn_q_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::attn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::qcur_reshaped, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::qcur_reshaped, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::qcur, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::reshape };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::qcur_rope, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::qcur_rope, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::qcur_reshaped, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::inp_pos, model_traits_type, kernel_type_profile_traits_type>;
		using input_type03 = core_traits<llama_op_types::rope_freqs_weight, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::rope };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	// KEY OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::kcur, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kcur, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::attn_k_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::attn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::kcur_reshaped, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kcur_reshaped, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kcur, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::reshape };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::kcur_rope, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kcur_rope, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kcur_reshaped, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::inp_pos, model_traits_type, kernel_type_profile_traits_type>;
		using input_type03 = core_traits<llama_op_types::rope_freqs_weight, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::rope };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	// VALUE OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::vcur, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::vcur, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::attn_v_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::attn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	// CACHE OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::k_cache_view, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::k_cache_view, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::cache_k, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::view };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::k_cache_view_copy, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::k_cache_view_copy, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kcur_rope, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::k_cache_view, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::copy };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::vcur_transposed, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::vcur_transposed, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::vcur, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::transpose };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::v_cache_view, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::v_cache_view, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::cache_v, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, model_traits_type::head_count_kv* model_traits_type::head_dim, 1, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::view };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::v_cache_view_copy, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::v_cache_view_copy, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::vcur_transposed, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::v_cache_view, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::kv_cache_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_count_kv * model_traits_type::head_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::copy };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	// ATTENTION COMPUTATION
	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::v, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::v, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::cache_v, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, model_traits_type::head_dim, model_traits_type::head_count_kv, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::view };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::k, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::k, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::cache_k, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::key_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::view };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::q, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::q, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::qcur_rope, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::query_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, 1, model_traits_type::head_count, model_traits_type::max_sequence_length } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::permute };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::kq, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kq, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::k, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::q, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::attention_score_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, 1, model_traits_type::head_count, model_traits_type::max_sequence_length } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::kq_soft_max, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kq_soft_max, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kq, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::kq_mask, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::softmax_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::max_sequence_length, 1, model_traits_type::head_count, model_traits_type::max_sequence_length } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::softmax };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::kqv, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kqv, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::v, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::kq_soft_max, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, 1, model_traits_type::head_count, model_traits_type::max_sequence_length } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::kqv_merged, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kqv_merged, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kqv, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::head_dim, model_traits_type::head_count, 1, model_traits_type::max_sequence_length } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::permute };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::kqv_merged_cont, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kqv_merged_cont, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kqv_merged, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::value_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ 0 };
		static constexpr kernel_type type{ kernel_type::cont };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::kqv_out, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::kqv_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::attn_output_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::kqv_merged_cont, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	// FFN OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::ffn_inp, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_inp, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::kqv_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::inp_embd, model_traits_type, kernel_type_profile_traits_type>;// or prev residual
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::add };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::norm_out, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::norm_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_inp, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::rms_norm };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_norm, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::norm_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::ffn_norm_weight, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_gate, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_gate, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_gate_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::ffn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_silu, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_silu, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_gate, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::silu };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::ffn_up, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_up, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_up_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::ffn_norm, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::ffn_gate_par, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_gate_par, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_silu, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::ffn_up, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::ffn_intermediate_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::ffn_out, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::ffn_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_down_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::ffn_gate_par, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::hidden_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type> struct core_traits<llama_op_types::l_out, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::l_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::ffn_out, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::ffn_inp, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::add };
		std::vector<cpu_core_thread<this_type>> threads{};
		array<output_type*, model_traits_type::block_count> data{};
	};

	// RESIDUAL CONNECTIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::attn_residual, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::attn_residual, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::inp_embd, model_traits_type, kernel_type_profile_traits_type>;// or prev layer output
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::get_rows };
		std::vector<cpu_core_thread<this_type>> threads{};
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::prev_residual, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::prev_residual, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::l_out, model_traits_type, kernel_type_profile_traits_type>;// from previous layer
		using output_type  = typename kernel_type_profile_traits_type::residual_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::get_rows };
		std::vector<cpu_core_thread<this_type>> threads{};
		output_type* data{};
	};

	// FINAL OUTPUT OPERATIONS
	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::final_norm, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::final_norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::l_out, model_traits_type, kernel_type_profile_traits_type>;// final layer output
		using output_type  = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::rms_norm };
		std::vector<cpu_core_thread<this_type>> threads{};
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::result_norm, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::result_norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::final_norm, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::output_norm_weight, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::norm_output_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul };
		std::vector<cpu_core_thread<this_type>> threads{};
		output_type* data{};
	};

	template<typename model_traits_type, typename kernel_type_profile_traits_type>
	struct core_traits<llama_op_types::result_output, model_traits_type, kernel_type_profile_traits_type> {
		RT_TM_FORCE_INLINE core_traits() noexcept							   = default;
		RT_TM_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		RT_TM_FORCE_INLINE core_traits(const core_traits&) noexcept			   = delete;
		RT_TM_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE core_traits(core_traits&&) noexcept				   = delete;

		using this_type	   = core_traits<llama_op_types::result_output, model_traits_type, kernel_type_profile_traits_type>;
		using input_type01 = core_traits<llama_op_types::output_weight, model_traits_type, kernel_type_profile_traits_type>;
		using input_type02 = core_traits<llama_op_types::result_norm, model_traits_type, kernel_type_profile_traits_type>;
		using output_type  = typename kernel_type_profile_traits_type::logit_type;
		static constexpr array<size_t, 4> dims{ { model_traits_type::vocab_size, model_traits_type::max_sequence_length, 1, 1 } };
		static constexpr size_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims) };
		static constexpr kernel_type type{ kernel_type::mul_mat };
		std::vector<cpu_core_thread<this_type>> threads{};
		output_type* data{};
		std::unique_ptr<std::latch> sync_flag_start{};
		std::unique_ptr<std::latch> sync_flag_end{};
	};

	template<typename op_type_type, typename model_traits_type, typename kernel_type_profile_traits_type> struct collect_required_bytes {
		template<op_type_type current_index = static_cast<op_type_type>(0)> RT_TM_FORCE_INLINE static constexpr size_t impl(size_t current_size = 0) {
			if constexpr (static_cast<size_t>(current_index) < static_cast<size_t>(op_type_type::count)) {
				current_size += core_traits<current_index, model_traits_type, kernel_type_profile_traits_type>::total_required_bytes;
				return impl<static_cast<op_type_type>(static_cast<size_t>(current_index) + 1)>(current_size);
			}
			return current_size;
		}
	};

	template<core_traits_type base_type> struct total_bytes_size<base_type> : public base_type {
		RT_TM_FORCE_INLINE total_bytes_size() noexcept									 = default;
		RT_TM_FORCE_INLINE total_bytes_size& operator=(const total_bytes_size&) noexcept = delete;
		RT_TM_FORCE_INLINE total_bytes_size(const total_bytes_size&) noexcept			 = delete;
		RT_TM_FORCE_INLINE total_bytes_size& operator=(total_bytes_size&&) noexcept		 = delete;
		RT_TM_FORCE_INLINE total_bytes_size(total_bytes_size&&) noexcept				 = delete;
		RT_TM_FORCE_INLINE static constexpr size_t impl() {
			if constexpr (array_type<decltype(base_type::data)>) {
				return base_type::total_required_bytes * decltype(base_type::data)::size_val;
			} else {
				return base_type::total_required_bytes;
			}
		}
	};

	template<typename base_type_new> struct thread_function {
		RT_TM_FORCE_INLINE thread_function() noexcept									   = default;
		RT_TM_FORCE_INLINE thread_function& operator=(const thread_function&) noexcept	   = delete;
		RT_TM_FORCE_INLINE thread_function(const thread_function&) noexcept				   = delete;
		RT_TM_FORCE_INLINE thread_function& operator=(thread_function&&) noexcept		   = delete;
		RT_TM_FORCE_INLINE thread_function(thread_function&&) noexcept					   = delete;
		using output_type																   = base_type_new::output_type;
		using base_type																	   = base_type_new;
		RT_TM_FORCE_INLINE static void impl(base_type& core, size_t thread_index) {
		}
	};

	template<blocking base_type_new> struct thread_function<base_type_new> {
		RT_TM_FORCE_INLINE thread_function() noexcept								   = default;
		RT_TM_FORCE_INLINE thread_function& operator=(const thread_function&) noexcept = delete;
		RT_TM_FORCE_INLINE thread_function(const thread_function&) noexcept			   = delete;
		RT_TM_FORCE_INLINE thread_function& operator=(thread_function&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE thread_function(thread_function&&) noexcept				   = delete;
		using output_type															   = base_type_new::output_type;
		using base_type																   = base_type_new;
		RT_TM_FORCE_INLINE static void impl(base_type& core, size_t thread_index) {
			core.sync_flag_start->arrive_and_wait();
			core.sync_flag_end->arrive_and_wait();
		}
	};

	template<typename base_type_new> struct execution_plammer {
		RT_TM_FORCE_INLINE execution_plammer() noexcept									   = default;
		RT_TM_FORCE_INLINE execution_plammer& operator=(const execution_plammer&) noexcept = delete;
		RT_TM_FORCE_INLINE execution_plammer(const execution_plammer&) noexcept			   = delete;
		RT_TM_FORCE_INLINE execution_plammer& operator=(execution_plammer&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE execution_plammer(execution_plammer&&) noexcept				   = delete;
		using output_type																   = base_type_new::output_type;
		using base_type																	   = base_type_new;
		RT_TM_FORCE_INLINE static void impl(base_type& core, size_t thread_count) {
			if constexpr (active_thread<base_type>) {
				for (size_t x = 0; x < thread_count; ++x) {
					cpu_core_thread<base_type_new> core_thread{ &core };
					core_thread.core		 = &core;
					core_thread.thread_index = x;
					core_thread.thread_count = thread_count;
					core.threads.emplace_back(core_thread);
				}
			}
		}
	};

	template<blocking base_type_new> struct execution_plammer<base_type_new> {
		RT_TM_FORCE_INLINE execution_plammer() noexcept									   = default;
		RT_TM_FORCE_INLINE execution_plammer& operator=(const execution_plammer&) noexcept = delete;
		RT_TM_FORCE_INLINE execution_plammer(const execution_plammer&) noexcept			   = delete;
		RT_TM_FORCE_INLINE execution_plammer& operator=(execution_plammer&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE execution_plammer(execution_plammer&&) noexcept				   = delete;
		using output_type																   = base_type_new::output_type;
		using base_type																	   = base_type_new;
		RT_TM_FORCE_INLINE static void impl(base_type& core, size_t thread_count) {
			core.sync_flag_start = std::make_unique<std::latch>(thread_count);
			core.sync_flag_end	 = std::make_unique<std::latch>(thread_count);
			for (size_t x = 0; x < thread_count; ++x) {
				cpu_core_thread<base_type_new> core_thread{ &core };
				core_thread.core			= &core;
				core_thread.thread_index	= x;
				core_thread.thread_count	= thread_count;
				core.threads.emplace_back(core_thread);
			}
		}
	};

	template<typename base_type> struct memory_mapper {
		RT_TM_FORCE_INLINE memory_mapper() noexcept								   = default;
		RT_TM_FORCE_INLINE memory_mapper& operator=(const memory_mapper&) noexcept = delete;
		RT_TM_FORCE_INLINE memory_mapper(const memory_mapper&) noexcept			   = delete;
		RT_TM_FORCE_INLINE memory_mapper& operator=(memory_mapper&&) noexcept	   = delete;
		RT_TM_FORCE_INLINE memory_mapper(memory_mapper&&) noexcept				   = delete;
		using output_type														   = base_type::output_type;
		template<typename memory_buffer_type> RT_TM_FORCE_INLINE static void impl(base_type& core, memory_buffer_type& memory_buffer) {
			if constexpr (base_type::total_required_bytes > 0) {
				output_type* ptr = static_cast<output_type*>(memory_buffer.claim_memory(core.total_required_bytes));
				if constexpr (array_type<decltype(core.data)>) {
					for (size_t x = 0; x < decltype(core.data)::size_val; ++x) {
						core.data[x] = ptr;
					}
				} else {
					core.data = ptr;
				}
			}
		}
	};

	template<typename... bases> struct core_bases : bases... {
		RT_TM_FORCE_INLINE core_bases() noexcept					= default;
		RT_TM_FORCE_INLINE core_bases& operator=(core_bases&&)		= delete;
		RT_TM_FORCE_INLINE core_bases(core_bases&&)					= delete;
		RT_TM_FORCE_INLINE core_bases& operator=(const core_bases&) = delete;
		RT_TM_FORCE_INLINE core_bases(const core_bases&)			= delete;

		template<template<typename> typename mixin_type, typename op_entity_type, typename... arg_types> RT_TM_FORCE_INLINE void impl_internal(arg_types&&... args) {
			return mixin_type<op_entity_type>::impl(*static_cast<op_entity_type*>(this), std::forward<arg_types>(args)...);
		}

		template<template<typename> typename mixin_type, typename... arg_types> RT_TM_FORCE_INLINE void impl(arg_types&&... args) {
			(impl_internal<mixin_type, bases>(args...), ...);
		}
	};

	template<typename op_type_type, typename model_traits_type, typename kernel_type_profile_traits_type, typename index_sequence> struct get_core_traits_base;

	template<typename op_type_type, typename model_traits_type, typename kernel_type_profile_traits_type, size_t... index>
	struct get_core_traits_base<op_type_type, model_traits_type, kernel_type_profile_traits_type, std::index_sequence<index...>> {
		using type = core_bases<core_traits<static_cast<op_type_type>(index), model_traits_type, kernel_type_profile_traits_type>...>;
	};

	template<typename op_type_type, typename model_traits_type, typename kernel_type_profile_traits_type> using get_core_traits_base_t =
		typename get_core_traits_base<op_type_type, model_traits_type, kernel_type_profile_traits_type, std::make_index_sequence<static_cast<size_t>(op_type_type::count)>>::type;

}
