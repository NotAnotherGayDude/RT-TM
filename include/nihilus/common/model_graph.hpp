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

#include <nihilus/common/memory_buffer.hpp>
#include <nihilus/common/core_base.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/model.hpp>

namespace nihilus {

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

	template<model_arch arch> struct construction_parameters;

	template<> struct construction_parameters<model_arch::llama> {
		uint64_t rope_dimension_count{};
		uint64_t feed_forward_length{};
		uint64_t embedding_length{};
		double f_attention_scale{};
		double rms_norm_epsilon{};
		double rope_attn_factor{};
		uint64_t context_length{};
		uint64_t n_expert_used{};
		uint64_t head_count_kv{};
		double rope_freq_scale{};
		double rope_ext_factor{};
		double rope_freq_base{};
		double rope_beta_fast{};
		double rope_beta_slow{};
		uint64_t block_count{};
		uint64_t head_count{};
		uint64_t vocab_size{};
		uint64_t rope_type{};
		uint64_t n_expert{};
	};

	template<model_config config> struct model_graph {
		NIHILUS_INLINE model_graph()								= default;
		NIHILUS_INLINE model_graph& operator=(model_graph&&)		= default;
		NIHILUS_INLINE model_graph(model_graph&&)					= default;
		NIHILUS_INLINE model_graph& operator=(const model_graph&) = delete;
		NIHILUS_INLINE model_graph(const model_graph&)			= delete;
		std::vector<core_base_creation_data> cores{};
		tokenizer_parameters<config.arch> tokenizer_params{};
		construction_parameters<config.arch> cparams{};
		hyper_parameters<config.arch> hparams{};
		memory_buffer<config> leaf_core_data{};
	};

}