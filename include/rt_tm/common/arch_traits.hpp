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

	template<model_arch arch> struct arch_traits {};

	template<> struct arch_traits<model_arch::llama> {
	  protected:
		static constexpr array llama_tensor_types{ tensor_type::input_tokens, tensor_type::token_embd, tensor_type::output_norm, tensor_type::output, tensor_type::rope_freqs,
			tensor_type::attn_norm, tensor_type::attn_q, tensor_type::attn_k, tensor_type::attn_v, tensor_type::attn_out, tensor_type::attn_rot_embd, tensor_type::ffn_gate_inp,
			tensor_type::ffn_norm, tensor_type::ffn_gate, tensor_type::ffn_down, tensor_type::ffn_up, tensor_type::ffn_gate_exp, tensor_type::ffn_down_exp, tensor_type::ffn_up_exp,
			tensor_type::ffn_gate_exps, tensor_type::ffn_down_exps, tensor_type::ffn_up_exps };

	  public:
		using enum_type = tensor_type;
		static constexpr auto tensor_types{ llama_tensor_types };
		static constexpr model_arch arch{ model_arch::llama };
		static constexpr size_t max_inputs{ 3 };
	};
}