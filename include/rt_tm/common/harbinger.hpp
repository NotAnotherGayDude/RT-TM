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

#include <rt_tm/common/model.hpp>
#include <rt_tm/common/model_parser.hpp>
#include <rt_tm/common/common.hpp>
#include <cstdint>

namespace rt_tm {

	struct harbinger {
		RT_TM_FORCE_INLINE static consteval auto generate_model_config(auto model_generation, auto model_size, kernel_type_profile kernel_profile, model_arch arch,
			bool exceptions = false, kv_cache_strategy cache_strategy = kv_cache_strategy::paged, bool use_gradient_checkpointing = false,
			rope_scaling_type rope_scaling = rope_scaling_type::linear, bool use_rotary_embeddings = true, uint64_t kv_cache_block_size = 16, bool use_flash_attention = true,
			norm_type rms_norm_type = norm_type::rms_standard, model_format format = model_format::gguf, float norm_epsilon = 1e-6f) {
			model_config<decltype(model_generation), decltype(model_size)> config{ model_generation, model_size, kernel_profile, arch, exceptions, cache_strategy,
				use_gradient_checkpointing, rope_scaling, use_rotary_embeddings, kv_cache_block_size, use_flash_attention, rms_norm_type, format, norm_epsilon };
			return config;
		};

		template<model_config config> RT_TM_FORCE_INLINE static auto parse_model_graph(std::string_view path) {
			if constexpr (cpu_arch_index == 2) {
				using model_type = model<impl_indices{ .cpu_index = 2 }, config>;
				using base_type	 = model_type::base_type;
				std::unique_ptr<base_type> return_value{};
				model_type* new_model{ new model_type{ path } };
				return_value.reset(new_model);
				return return_value;
			} else if constexpr (cpu_arch_index == 1) {
				using model_type = model<impl_indices{ .cpu_index = 1 }, config>;
				using base_type	 = model_type::base_type;
				std::unique_ptr<base_type> return_value{};
				model_type* new_model{ new model_type{ path } };
				return_value.reset(new_model);
				return return_value;
			} else {
				using model_type = model<impl_indices{ .cpu_index = 0 }, config>;
				using base_type	 = model_type::base_type;
				std::unique_ptr<base_type> return_value{};
				model_type* new_model{ new model_type{ path } };
				return_value.reset(new_model);
				return return_value;
			}
		}

		RT_TM_FORCE_INLINE static cli_params parse_cli_arguments(const std::string& command_line) {
			cli_params result{};
			std::istringstream stream(command_line);
			std::string current_flag{};
			std::string token{};
			bool expect_value = false;

			while (stream >> std::quoted(token) || stream >> token) {
				if (token.empty())
					continue;

				if (token[0] == '-') {
					current_flag = token;
					if (token == "-m" || token == "-t") {
						expect_value = true;
					} else {
						expect_value = false;
					}
				} else if (expect_value) {
					if (current_flag == "-m") {
						result.model_file = token;
					} else if (current_flag == "-t") {
						try {
							result.thread_count = std::stoull(token);
						} catch (const std::exception&) {
							result.thread_count = 1;
						}
					}
					expect_value = false;
				}
			}

			return result;
		}
	};

}
