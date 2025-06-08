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

#include <rt_tm/common/memory_buffer.hpp>
#include <rt_tm/common/core_base.hpp>
#include <rt_tm/common/common.hpp>
#include <rt_tm/common/block.hpp>

namespace rt_tm {

	template<model_arch arch> struct tokenizer_parameters;

	template<> struct tokenizer_parameters<model_arch::llama> {
		std::vector<int64_t> token_types{};
		std::vector<std::string> tokens{};
		std::vector<std::string> merges{};
		std::string chat_template{};
		uint64_t bos_token_id{};
		uint64_t eos_token_id{};
		std::string pre{};
	};

	enum class llama_construction_parameter_type_uint64 {
		rope_dimension_count,
		feed_forward_length,
		embedding_length,
		context_length,
		n_expert_used,
		head_count_kv,
		block_count,
		head_count,
		vocab_size,
		rope_type,
		n_expert,
		count,
	};

	enum class llama_construction_parameter_type_float64 {
		f_attention_scale,
		rms_norm_epsilon,
		rope_attn_factor,
		rope_freq_scale,
		rope_ext_factor,
		rope_freq_base,
		rope_beta_fast,
		rope_beta_slow,
		count,
	};

	enum class llama_hyper_parameter_type_uint64 {
		current_sequence_length,
		kv_cache_size_per_layer,
		batch_size,
		rope_dims,
		count,
	};

	enum class llama_hyper_parameter_type_float64 {
		rope_freqs,
		count,
	};

	template<model_arch arch> struct construction_parameters;

	template<> struct construction_parameters<model_arch::llama> {
		uint64_t params_uint64[static_cast<size_t>(llama_construction_parameter_type_uint64::count)]{};
		double params_float64[static_cast<size_t>(llama_construction_parameter_type_float64::count)]{};
		RT_TM_FORCE_INLINE uint64_t& operator()(llama_construction_parameter_type_uint64 index) {
			return params_uint64[static_cast<size_t>(index)];
		}

		RT_TM_FORCE_INLINE double& operator()(llama_construction_parameter_type_float64 index) {
			return params_float64[static_cast<size_t>(index)];
		}

		RT_TM_FORCE_INLINE const uint64_t& operator()(llama_construction_parameter_type_uint64 index) const {
			return params_uint64[static_cast<size_t>(index)];
		}

		RT_TM_FORCE_INLINE const double& operator()(llama_construction_parameter_type_float64 index) const {
			return params_float64[static_cast<size_t>(index)];
		}
	};

	template<model_arch arch> struct hyper_parameters;

	template<> struct hyper_parameters<model_arch::llama> {
		uint64_t params_uint64[static_cast<size_t>(llama_hyper_parameter_type_uint64::count)]{};
		double params_float64[static_cast<size_t>(llama_hyper_parameter_type_float64::count)]{};
		RT_TM_FORCE_INLINE uint64_t& operator()(llama_hyper_parameter_type_uint64 index) {
			return params_uint64[static_cast<size_t>(index)];
		}

		RT_TM_FORCE_INLINE double& operator()(llama_hyper_parameter_type_float64 index) {
			return params_float64[static_cast<size_t>(index)];
		}

		RT_TM_FORCE_INLINE const uint64_t& operator()(llama_hyper_parameter_type_uint64 index) const {
			return params_uint64[static_cast<size_t>(index)];
		}

		RT_TM_FORCE_INLINE const double& operator()(llama_hyper_parameter_type_float64 index) const {
			return params_float64[static_cast<size_t>(index)];
		}
	};

	template<global_config config> struct model_graph {
		RT_TM_INLINE model_graph()								= default;
		RT_TM_INLINE model_graph& operator=(model_graph&&)		= default;
		RT_TM_INLINE model_graph(model_graph&&)					= default;
		RT_TM_INLINE model_graph& operator=(const model_graph&) = delete;
		RT_TM_INLINE model_graph(const model_graph&)			= delete;
		std::vector<core_base_creation_data> op_cores{};
		tokenizer_parameters<config.arch> tokenizer_params{};
		construction_parameters<config.arch> cparams{};
		hyper_parameters<config.arch> hparams{};
		memory_buffer<config> leaf_core_data{};
	};

}