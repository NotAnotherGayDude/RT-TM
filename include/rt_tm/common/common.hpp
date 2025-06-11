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

#include <rt_tm/common/string_literal.hpp>
#include <rt_tm/common/data_types.hpp>
#include <rt_tm/common/concepts.hpp>
#include <iostream>
#include <cstdint>
#include <thread>
#include <mutex>

namespace rt_tm {

	inline std::mutex mutex{};

	RT_TM_FORCE_INLINE void log(std::string_view string) {
		std::unique_lock lock{ mutex };
		std::cout << string << std::endl;
	}

	// from
	// https://stackoverflow.com/questions/16337610/how-to-know-if-a-type-is-a-specialization-of-stdvector
	template<typename, template<typename...> typename> constexpr bool is_specialization_v = false;

	template<template<typename...> typename value_type, typename... arg_types> constexpr bool is_specialization_v<value_type<arg_types...>, value_type> = true;

	template<size_t N, auto lambda, typename... args> inline constexpr void for_each(args&&... arg_vals) {
		if constexpr (N > 0) {
			[&]<size_t... I>(std::index_sequence<I...>) {
				( void )(lambda.template operator()<I>(std::forward<args>(arg_vals)...), ...);
			}(std::make_index_sequence<N>{});
		}
	}

	template<auto current_index, auto enum_count> RT_TM_FORCE_INLINE constexpr std::string_view get_enum_name() {
		std::string_view return_string{ std::source_location::current().function_name() };
		auto new_size	 = std::size("get_enum_name<");
		size_t new_index = return_string.find("get_enum_name<") + new_size - 1;
		return_string	 = return_string.substr(new_index, return_string.size() - new_index);
		return_string	 = return_string.substr(0, return_string.find(','));
		return return_string;
	}

	template<auto current_index, auto enum_count> RT_TM_FORCE_INLINE std::string print_enum_value(auto enum_val) {
		if constexpr (static_cast<size_t>(current_index) < static_cast<size_t>(enum_count)) {
			if (static_cast<size_t>(current_index) == static_cast<size_t>(enum_val)) {
				constexpr std::string_view string{ get_enum_name<current_index, enum_count>() };
				return static_cast<std::string>(string);
			} else {
				return print_enum_value<static_cast<decltype(enum_count)>(static_cast<size_t>(current_index) + 1), enum_count>(enum_val);
			}
		} else {
			return {};
		}
	};

	enum class data_type : uint64_t {
		f32	 = 0,
		f16	 = 1,
		q8_0 = 8,
		i8	 = 24,
		i16	 = 25,
		i32	 = 26,
		i64	 = 27,
		f64	 = 28,
		count,
	};

	enum class kernel_type : uint8_t {
		none,
		get_rows,
		rms_norm,
		mul,
		mul_mat,
		reshape,
		permute,
		transpose,
		view,
		cont,
		copy,
		rope,
		softmax,
		silu,
		add,
		sub,
	};

	enum class llama_op_types : uint16_t {
		inp_embd,
		token_embd_weight,
		inp_tokens,
		inp_pos,
		inp_out_ids,
		rope_freqs_weight,
		output_weight,
		output_norm_weight,
		attn_q_weight,
		attn_k_weight,
		attn_v_weight,
		attn_output_weight,
		attn_norm_weight,
		ffn_gate_weight,
		ffn_up_weight,
		ffn_down_weight,
		ffn_norm_weight,
		cache_k,
		cache_v,
		kq_mask,
		norm,
		attn_norm,
		qcur,
		qcur_reshaped,
		qcur_rope,
		kcur,
		kcur_reshaped,
		kcur_rope,
		vcur,
		k_cache_view,
		k_cache_view_copy,
		vcur_transposed,
		v_cache_view,
		v_cache_view_copy,
		v,
		k,
		q,
		kq,
		kq_soft_max,
		kqv,
		kqv_merged,
		kqv_merged_cont,
		kqv_out,
		ffn_inp,
		norm_out,
		ffn_norm,
		ffn_gate,
		ffn_silu,
		ffn_up,
		ffn_gate_par,
		ffn_out,
		l_out,
		attn_residual,
		prev_residual,
		final_norm,
		result_norm,
		result_output,
		count
	};

	enum class device_type {
		cpu,
		gpu,
		numa,
	};

	enum class model_arch {
		llama,
		count,
	};

	enum class kernel_type_profile : size_t {
		fp16_mha,
		fp16_moe,
		bf16_mha,
		bf16_gqa,
		q4_mha,
		q4_gqa,
		q4_moe,
		q8_mha,
		q8_gqa,
		q8_moe,
		mixed_fp16_fp32,
		mixed_bf16_fp32,
		count,
	};

	enum class norm_type : size_t {
		rms_standard,
		rms_parallel,
		rms_grouped,
		layer_norm_standard,
		layer_norm_no_bias,
		rms_norm_welford,
		adaptive_norm,
		count,
	};

	enum class kv_cache_strategy : size_t {
		contiguous,
		paged,
		compressed,
		streaming,
		hierarchical,
		count,
	};

	enum class rope_scaling_type : size_t {
		none,
		linear,
		dynamic,
		yarn,
		longrope,
		count,
	};

	template<model_arch arch> struct op_type_type;

	template<> struct op_type_type<model_arch::llama> {
		using type = llama_op_types;
	};

	template<typename model_generation_type_new, typename model_size_type_new> struct model_config {
		using model_generation_type = model_generation_type_new;
		using model_size_type		= model_size_type_new;
		model_generation_type model_generation{};
		model_size_type model_size{};
		kernel_type_profile kernel_profile{};
		model_arch arch{};
		kv_cache_strategy cache_strategy{};
		bool use_gradient_checkpointing{};
		rope_scaling_type rope_scaling{};
		bool use_rotary_embeddings{};
		size_t kv_cache_block_size{};
		bool use_flash_attention{};
		norm_type rms_norm_type{};
		float norm_epsilon{};
		bool exceptions{};

	  protected:
		template<typename model_generateion_type_newer, typename model_size_type_newer> friend struct model_base;
		friend consteval auto generate_model_config(auto model_generation, auto model_size, kernel_type_profile kernel_profile, model_arch arch, bool exceptions,
			kv_cache_strategy cache_strategy, bool use_gradient_checkpointing, rope_scaling_type rope_scaling, bool use_rotary_embeddings, size_t kv_cache_block_size,
			bool use_flash_attention, norm_type rms_norm_type, float norm_epsilon);

		constexpr model_config(auto model_generation_new, auto model_size_new, kernel_type_profile kernel_profile_new, model_arch arch_new, kv_cache_strategy cache_strategy_new,
			bool use_gradient_checkpointing_new, rope_scaling_type rope_scaling_new, bool use_rotary_embeddings_new, size_t kv_cache_block_size_new, bool use_flash_attention_new,
			norm_type rms_norm_type_new, float norm_epsilon_new, bool exceptions_new)
			: model_generation(model_generation_new), model_size(model_size_new), kernel_profile(kernel_profile_new), arch(arch_new), cache_strategy(cache_strategy_new),
			  use_gradient_checkpointing(use_gradient_checkpointing_new), rope_scaling(rope_scaling_new), use_rotary_embeddings(use_rotary_embeddings_new),
			  kv_cache_block_size(kv_cache_block_size_new), use_flash_attention(use_flash_attention_new), rms_norm_type(rms_norm_type_new), norm_epsilon(norm_epsilon_new),
			  exceptions(exceptions_new) {};

		constexpr model_config() = default;
	};

	struct cli_params {
		std::string model_file{};
		size_t thread_count{};
	};

	struct impl_indices {
		size_t cpu_index{};
		size_t gpu_index{};
	};

	struct op_graph_config {
		size_t num_threads{ std::thread::hardware_concurrency() };
	};
}
