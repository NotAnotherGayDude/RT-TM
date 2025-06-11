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
#include <rt_tm/common/tuple.hpp>

namespace rt_tm {

	template<typename weight_type_new, typename activation_type_new, typename compute_type_new, typename scale_type_new, typename index_type_new, typename output_type_new>
	struct kernel_type_profile_traits_impl {
		using weight_type			 = weight_type_new;
		using activation_type		 = activation_type_new;
		using compute_type			 = compute_type_new;
		using scale_type			 = scale_type_new;
		using index_type			 = index_type_new;
		using output_type			 = output_type_new;
		using input_token_type		 = index_type;
		using position_type			 = index_type;
		using output_token_type		 = index_type;
		using bias_type				 = activation_type;
		using zero_point_type		 = weight_type;
		using expert_weight_type	 = weight_type;
		using norm_weight_type		 = activation_type;
		using embedding_type		 = activation_type;
		using hidden_type			 = activation_type;
		using residual_type			 = activation_type;
		using query_type			 = activation_type;
		using key_type				 = activation_type;
		using value_type			 = activation_type;
		using kv_cache_type			 = activation_type;
		using ffn_intermediate_type	 = activation_type;
		using ffn_gate_type			 = activation_type;
		using ffn_up_type			 = activation_type;
		using ffn_down_type			 = activation_type;
		using expert_activation_type = activation_type;
		using expert_output_type	 = activation_type;
		using attention_weight_type	 = activation_type;
		using norm_output_type		 = activation_type;
		using rope_freq_type		 = activation_type;
		using rope_cos_type			 = activation_type;
		using rope_sin_type			 = activation_type;
		using accumulator_type		 = compute_type;
		using intermediate_type		 = compute_type;
		using attention_score_type	 = compute_type;
		using softmax_type			 = compute_type;
		using reduction_type		 = compute_type;
		using norm_type				 = compute_type;
		using routing_logit_type	 = compute_type;
		using loss_type				 = compute_type;
		using gradient_type			 = compute_type;
		using logit_type			 = output_type;
		using probability_type		 = output_type;
		using routing_weight_type	 = scale_type;
		using offset_type			 = std::make_signed_t<index_type>;
		using stride_type			 = index_type;
		using attention_mask_type	 = activation_type;
	};

	template<kernel_type_profile kernel_profile> struct kernel_type_profile_traits;

	template<> struct kernel_type_profile_traits<kernel_type_profile::q8_gqa> : public kernel_type_profile_traits_impl<int8_t, uint16_t, float, uint16_t, int32_t, float> {};

}
