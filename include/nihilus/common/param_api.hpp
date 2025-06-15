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

#include <nihilus/common/common.hpp>

namespace nihilus {

	enum class rope_aux_params : uint64_t {
		rope_dimension_count = 0,
		rope_freq_base		 = 8,
		freq_base			 = 16,
		freq_scale			 = 24,
		ext_factor			 = 32,
		attn_factor			 = 40,
		beta_fast			 = 48,
		beta_slow			 = 56,
		sections			 = 64,
		count				 = 96,
	};

	enum class rms_norm_aux_params : uint64_t {
		rms_norm_epsilon = 0,
		count			 = 8,
	};

	enum class attention_aux_params : uint64_t {
		head_count		 = 0,
		head_count_kv	 = 8,
		embedding_length = 16,
		count			 = 24,
	};

	enum class ffn_aux_params : uint64_t {
		feed_forward_length = 0,
		embedding_length	= 8,
		count				= 16,
	};

	template<typename derived_type> struct param_api {
		template<typename value_type, auto offset> NIHILUS_FORCE_INLINE const value_type& get_value() const {
			const value_type* ptr = static_cast<const value_type*>(static_cast<const void*>(static_cast<const derived_type*>(this)->aux_params.data()));
			return ptr[static_cast<uint64_t>(offset) / sizeof(value_type)];
		}

		template<typename value_type, auto offset> NIHILUS_FORCE_INLINE value_type& get_value() {
			value_type* ptr = static_cast<value_type*>(static_cast<void*>(static_cast<derived_type*>(this)->aux_params.data()));
			return ptr[static_cast<uint64_t>(offset) / sizeof(value_type)];
		}

		template<auto offset, typename value_type> NIHILUS_FORCE_INLINE void set_value(value_type value) {
			static_cast<derived_type*>(this)->aux_params.resize(static_cast<derived_type*>(this)->aux_params.size() + sizeof(value_type));
			std::memcpy(static_cast<derived_type*>(this)->aux_params.data() + static_cast<uint64_t>(offset), &value, sizeof(value));
		}
	};

}