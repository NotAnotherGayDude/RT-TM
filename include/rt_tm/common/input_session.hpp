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

#include <rt_tm/common/tokenizer.hpp>
#include <rt_tm/common/config.hpp>
#include <iterator>

namespace rt_tm {

	struct input_session_config {
		RT_TM_FORCE_INLINE input_session_config& operator=(const input_session_config&) = delete;
		RT_TM_FORCE_INLINE input_session_config(const input_session_config&)			= delete;
		RT_TM_FORCE_INLINE input_session_config(std::istream& stream_new, uint64_t max_tokens_new) : stream{ stream_new }, max_tokens{ max_tokens_new } {};
		std::istream& stream;
		uint64_t max_tokens{};
	};

	template<typename model_type> struct input_session : public tokenizer<model_type::model_traits_type::arch> {
		RT_TM_FORCE_INLINE input_session(const input_session_config&, model_type& model) : model_ptr{ &model } {};

		RT_TM_FORCE_INLINE bool process_input() {
			this->tokenize(input, model_ptr->template get_core<model_type::op_type_type::inp_tokens>().data);
			model_ptr->execute_tasks();
			return false;
		}

		RT_TM_FORCE_INLINE operator bool() {
			return false;
		}

	  protected:
		model_type* model_ptr{};
		std::string input{};
	};

}
