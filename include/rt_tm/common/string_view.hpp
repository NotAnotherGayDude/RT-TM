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
#include <rt_tm/common/common.hpp>

namespace rt_tm {

	struct string_view {
		const char* data_val{};
		size_t size_val{};

		RT_TM_FORCE_INLINE string_view() noexcept = default;

		RT_TM_FORCE_INLINE string_view& operator=(string_literal other) {
			size_val = other.size();
			data_val = other.data();
			return *this;
		}

		RT_TM_FORCE_INLINE string_view(string_literal other) {
			*this = other;
		}

		RT_TM_FORCE_INLINE size_t size() const {
			return size_val;
		}

		RT_TM_FORCE_INLINE const char* data() const {
			return data_val;
		}

		RT_TM_FORCE_INLINE operator std::string() const {
			return { data_val, size_val };
		}

		template<typename value_type> RT_TM_FORCE_INLINE bool operator==(const value_type& other) const {
			if (size() != other.size()) {
				return false;
			}
			return std::memcmp(other.data(), data(), size()) == 0;
		};
	};

}