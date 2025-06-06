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

#include <rt_tm/common/config.hpp>
#include <iterator>

namespace rt_tm {

	struct string_literal {
		constexpr string_literal() noexcept = default;
		constexpr string_literal(const std::string_view& string) noexcept {
			std::copy(string.data(), string.data() + string.size(), data_val);
			size_val = string.size() > 0 ? string.size() : 0;
		};

		template<size_t size_new> constexpr string_literal(const char (&string)[size_new]) noexcept {
			std::copy(string, string + size_new, data_val);
			size_val = size_new > 0 ? size_new - 1 : 0;
		};
		constexpr operator const char*() const {
			return data_val;
		}

		constexpr const char* data() const {
			return data_val;
		}

		constexpr char* data() {
			return data_val;
		}

		constexpr size_t size() const {
			return size_val;
		}
		size_t size_val{};

		constexpr auto operator+=(const string_literal& str) const noexcept {
			string_literal newLiteral{};
			newLiteral.size_val = str.size() + size();
			std::copy(data_val, data_val + size(), newLiteral.data());
			std::copy(str.data(), str.data() + str.size(), newLiteral.data() + size());
			return newLiteral;
		}

		constexpr auto operator+(const string_literal& str) const noexcept {
			string_literal newLiteral{};
			newLiteral.size_val = str.size() + size();
			std::copy(data_val, data_val + size(), newLiteral.data());
			std::copy(str.data(), str.data() + str.size(), newLiteral.data() + size());
			return newLiteral;
		}

		char data_val[56]{};
	};

	template<typename value_type> constexpr uint64_t count_digits(value_type number) noexcept {
		uint64_t count = 0;
		if (static_cast<int64_t>(number) < 0) {
			number *= -1;
			++count;
		}
		do {
			++count;
			number /= 10;
		} while (number != 0);
		return count;
	}

	template<auto number, size_t numDigits = count_digits(number)> constexpr string_literal to_string_literal() noexcept {
		char buffer[numDigits + 1]{};
		char* ptr = buffer + numDigits;
		*ptr	  = '\0';
		int64_t temp{};
		if constexpr (number < 0) {
			temp			   = number * -1;
			*(ptr - numDigits) = '-';
		} else {
			temp = number;
		}
		do {
			*--ptr = '0' + (temp % 10);
			temp /= 10;
		} while (temp != 0);
		return string_literal{ buffer };
	}

	template<size_t index> constexpr string_literal replace_format_with_index(const string_literal& input) {
		string_literal result;
		constexpr auto index_str = to_string_literal<index>();

		size_t format_pos = std::numeric_limits<size_t>::max();
		for (size_t i = 0; i < input.size() - 1; ++i) {
			if (input.data_val[i] == '%' && input.data_val[i + 1] == 'd') {
				format_pos = i;
				break;
			}
		}

		if (format_pos == std::numeric_limits<size_t>::max()) {
			return input;
		}

		size_t result_pos = 0;

		for (size_t i = 0; i < format_pos; ++i) {
			result.data_val[result_pos++] = input.data_val[i];
		}

		for (size_t i = 0; i < index_str.size(); ++i) {
			result.data_val[result_pos++] = index_str.data_val[i];
		}

		for (size_t i = format_pos + 2; i < input.size(); ++i) {
			result.data_val[result_pos++] = input.data_val[i];
		}

		result.size_val				= result_pos;
		result.data_val[result_pos] = '\0';

		return result;
	}

	constexpr bool has_format_specifier(const string_literal& base_name) {
		for (size_t i = 0; i < base_name.size() - 1; ++i) {
			if (base_name.data_val[i] == '%' && base_name.data_val[i + 1] == 'd') {
				return true;
			}
		}
		return false;
	}

	template<size_t max_index, size_t current_index = 0> constexpr void fill_indexed_array(array<string_literal, max_index>& result, const string_literal& base_name) {
		if constexpr (current_index < max_index) {
			result[current_index] = replace_format_with_index<current_index>(base_name);
			fill_indexed_array<max_index, current_index + 1>(result, base_name);
		}
	}

	template<size_t max_index> constexpr void fill_single_enum_array(array<string_literal, max_index>& result, size_t enum_index, const auto& string_array) {
		const string_literal base_name{ string_array[enum_index] };
		if (has_format_specifier(base_name)) {
			fill_indexed_array<max_index>(result, base_name);
		} else {
			for (size_t i = 0; i < max_index; ++i) {
				result[i] = base_name;
			}
		}
	}

	template<size_t max_index, size_t enum_count, auto current_enum_index = 0>
	constexpr void fill_all_enum_arrays(array<array<string_literal, max_index>, enum_count>& result, const auto& string_array) {
		if constexpr (current_enum_index < enum_count) {
			fill_single_enum_array<max_index>(result[current_enum_index], current_enum_index, string_array);
			if constexpr (current_enum_index + 1 < enum_count) {
				fill_all_enum_arrays<max_index, enum_count, current_enum_index + 1>(result, string_array);
			}
		}
	}

	template<size_t max_index, size_t enum_count> constexpr array<array<string_literal, max_index>, enum_count> generate_all_enum_arrays(const auto& string_array) {
		array<array<string_literal, max_index>, enum_count> result = {};
		fill_all_enum_arrays<max_index, enum_count>(result, string_array);
		return result;
	}
}