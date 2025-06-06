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

#include <rt_tm/common/core_base.hpp>
#include <rt_tm/common/memory_buffer.hpp>
#include <rt_tm/common/common.hpp>

namespace rt_tm {

	enum class llama_op_names {
		token_embed,
		rope_freqs,
		rope_dims,
		output_norm,
		blk_output,
		blk_attn_q,
		blk_attn_norm,
		blk_attn_k,
		blk_attn_v,
		blk_ffn_down,
		blk_ffn_gate,
		blk_attn_output,
		blk_ffn_norm,
		blk_ffn_up,
		cache_k,
		cache_v,
		input_tokens,
		input_embedding,
		norm,
		attn_norm,
		q_proj,
		q_reshape,
		rope_q,
		k_proj,
		k_reshape,
		rope_k,
		v_proj,
		v_reshape,
		k_cache_view,
		k_cache_copy,
		v_reshape_2,
		v_transpose,
		v_cache_view,
		v_cache_copy,
		v_cache_view_2,
		v_cache_permute,
		k_cache_view_2,
		k_cache_permute,
		q_permute,
		attn_scores,
		attn_weights,
		attn_out,
		attn_permute,
		attn_cont,
		attn_proj,
		residual_add,
		ffn_norm,
		ffn_norm_mul,
		ffn_gate,
		ffn_silu,
		ffn_up,
		ffn_gate_mul,
		ffn_down,
		layer_out,
		op_count,
		count
	};

	template<model_arch arch> struct model_arch_traits {};

	template<> struct model_arch_traits<model_arch::llama> {
	  protected:
		static constexpr array llama_op_name_strings{ string_literal{ "token_embd.weight" }, "rope_freqs.weight", "rope_dims.weight", "output_norm.weight", "output.weight",
			"blk.%d.attn_q.weight", "blk.%d.attn_norm.weight", "blk.%d.attn_k.weight", "blk.%d.attn_v.weight", "blk.%d.ffn_down.weight", "blk.%d.ffn_gate.weight",
			"blk.%d.attn_output.weight", "blk.%d.ffn_norm.weight", "blk.%d.ffn_up.weight", "cache_k_l%d", "cache_v_l%d", "inp_tokens", "inp_embd", "norm-%d", "attn_norm-%d",
			"q_proj-%d", "q_reshape-%d", "rope_q-%d", "k_proj-%d", "k_reshape-%d", "rope_k-%d", "v_proj-%d", "v_reshape-%d", "k_cache_view-%d", "k_cache_copy-%d", "v_reshape_2-%d",
			"v_transpose-%d", "v_cache_view-%d", "v_cache_copy-%d", "v_cache_view_2-%d", "v_cache_permute-%d", "k_cache_view_2-%d", "k_cache_permute-%d", "q_permute-%d",
			"attn_scores-%d", "attn_weights-%d", "attn_out-%d", "attn_permute-%d", "attn_cont-%d", "attn_proj-%d", "residual_add-%d", "ffn_norm-%d", "ffn_norm_mul-%d",
			"ffn_gate-%d", "ffn_silu-%d", "ffn_up-%d", "ffn_gate_mul-%d", "ffn_down-%d", "layer_out-%d", "logits_output" };

	  public:
		using enum_type = llama_op_names;
		static constexpr auto op_names{ generate_all_enum_arrays<32, static_cast<size_t>(llama_op_names::count)>(llama_op_name_strings) };
		static constexpr size_t max_inputs{ 3 };
	};

}