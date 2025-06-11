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

	template<model_config config> struct harbinger {
		template<model_format format> RT_TM_FORCE_INLINE static auto parse_model_graph(std::string_view path) {
			//return std::make_unique<model_base<typename model_config::model_generation_type, typename model_config::model_size_type>>(model<model_config>{});
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
