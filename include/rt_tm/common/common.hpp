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
#include <rt_tm/common/concepts.hpp>
#include <cstdint>
#include <thread>

namespace rt_tm {

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

	enum class op_type : uint64_t {
		unset	  = 0,
		noop	  = 1,
		mul_mat	  = 2,
		mul		  = 3,
		add		  = 4,
		sub		  = 5,
		get_rows  = 6,
		view	  = 7,
		copy	  = 8,
		softmax	  = 9,
		rms_norm  = 10,
		reshape	  = 11,
		rope	  = 12,
		transpose = 13,
		permute	  = 14,
		cont	  = 15,
		silu	  = 16,
		count,
	};

	enum class tensor_type : uint64_t {
		token_embd,
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
		cpu = 0,
		gpu = 1,
	};

	enum class model_arch {
		llama,
		count,
	};

	struct global_config {
		model_arch arch{};
		bool exceptions{};
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
