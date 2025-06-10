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
#include <cstdint>
#include <thread>

namespace rt_tm {

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
		float_32 = 0,
		float_16 = 1,
		q8_0	 = 8,
		int_8	 = 24,
		int_16	 = 25,
		int_32	 = 26,
		int_64	 = 27,
		float_64 = 28,
		count,
	};

	enum class kernel_type : uint64_t {
		unset,
		mul_mat,
		mul,
		add,
		sub,
		get_rows,
		view,
		copy,
		softmax,
		rms_norm,
		reshape,
		rope,
		transpose,
		permute,
		cont,
		silu,
		noop,
		count,
	};

	enum class op_types : uint64_t {
		KQ_mask,
		Kcur,
		Qcur,
		Vcur,
		attn_norm,
		attn_k,
		attn_output,
		attn_q,
		attn_v,
		ffn_down,
		ffn_gate,
		ffn_norm,
		ffn_up,
		cache_k_l,
		ffn_gate_par,
		ffn_inp,
		ffn_out,
		ffn_silu,
		inp_embd,
		inp_out_ids,
		inp_pos,
		inp_tokens,
		k,
		k_cache_view,
		kq,
		kq_soft_max_ext,
		kqv,
		kqv_merged,
		kqv_merged_cont,
		kqv_out,
		l_out,
		node_1016,
		node_1017,
		norm,
		output,
		output_norm,
		q,
		result_norm,
		result_output,
		rope_freqs,
		token_embd,
		v,
		count
	};

	enum class tensor_type : uint64_t {
		input_embed,
		input_tokens,
		token_position,
		token_embd,
		k_cache,
		v_cache,
		token_embd_norm,
		token_types,
		pos_embd,
		output,
		output_norm,
		rope_freqs,
		rope_factors_long,
		rope_factors_short,
		attn_q,
		attn_k,
		attn_v,
		attn_qkv,
		attn_out,
		attn_norm,
		attn_norm_2,
		attn_out_norm,
		attn_post_norm,
		attn_rot_embd,
		ffn_gate_inp,
		ffn_gate_inp_shexp,
		ffn_norm,
		ffn_post_norm,
		ffn_gate,
		ffn_down,
		ffn_up,
		ffn_act,
		ffn_down_exp,
		ffn_gate_exp,
		ffn_up_exp,
		ffn_norm_exps,
		ffn_down_exps,
		ffn_gate_exps,
		ffn_up_exps,
		ffn_down_shexp,
		ffn_gate_shexp,
		ffn_up_shexp,
		ffn_exp_probs_b,
		attn_q_norm,
		attn_k_norm,
		layer_out_norm,
		ssm_in,
		ssm_conv1d,
		ssm_x,
		ssm_dt,
		ssm_a,
		ssm_d,
		ssm_out,
		time_mix_w1,
		time_mix_w2,
		time_mix_lerp_x,
		time_mix_lerp_w,
		time_mix_lerp_k,
		time_mix_lerp_v,
		time_mix_lerp_r,
		time_mix_lerp_g,
		time_mix_lerp_fused,
		time_mix_first,
		time_mix_decay,
		time_mix_decay_w1,
		time_mix_decay_w2,
		time_mix_key,
		time_mix_value,
		time_mix_receptance,
		time_mix_gate,
		time_mix_ln,
		time_mix_output,
		channel_mix_lerp_k,
		channel_mix_lerp_r,
		channel_mix_key,
		channel_mix_receptance,
		channel_mix_value,
		attn_q_a,
		attn_q_b,
		attn_kv_a_mqa,
		attn_kv_b,
		attn_q_a_norm,
		attn_kv_a_norm,
		attn_sub_norm,
		ffn_sub_norm,
		dec_attn_norm,
		dec_attn_q,
		dec_attn_k,
		dec_attn_v,
		dec_attn_out,
		dec_attn_rel_b,
		dec_cross_attn_norm,
		dec_cross_attn_q,
		dec_cross_attn_k,
		dec_cross_attn_v,
		dec_cross_attn_out,
		dec_cross_attn_rel_b,
		dec_ffn_norm,
		dec_ffn_gate,
		dec_ffn_down,
		dec_ffn_up,
		dec_output_norm,
		enc_attn_norm,
		enc_attn_q,
		enc_attn_k,
		enc_attn_v,
		enc_attn_out,
		enc_attn_rel_b,
		enc_ffn_norm,
		enc_ffn_gate,
		enc_ffn_down,
		enc_ffn_up,
		enc_output_norm,
		cls,
		cls_out,
		conv1d,
		convnext_dw,
		convnext_norm,
		convnext_pw1,
		convnext_pw2,
		convnext_gamma,
		pos_net_conv1,
		pos_net_conv2,
		pos_net_norm,
		pos_net_norm1,
		pos_net_norm2,
		pos_net_attn_norm,
		pos_net_attn_q,
		pos_net_attn_k,
		pos_net_attn_v,
		pos_net_attn_out,
		count
	};

	enum class device_type {
		cpu	 = 0,
		gpu	 = 1,
		numa = 2,
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

	template<typename model_generation_type, typename model_size_type> struct model_config {
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
		template<typename model_generation_type_new, typename model_size_type_new> friend struct model_base;
		friend consteval auto generate_model_config(auto model_generation, auto model_size, kernel_type_profile kernel_profile, model_arch arch, kv_cache_strategy cache_strategy,
			bool use_gradient_checkpointing, rope_scaling_type rope_scaling, bool use_rotary_embeddings, size_t kv_cache_block_size, bool use_flash_attention,
			norm_type rms_norm_type, float norm_epsilon, bool exceptions);

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
