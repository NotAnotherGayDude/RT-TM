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

#include <rt_tm/common/type_traits.hpp>
#include <rt_tm/common/model_graph.hpp>
#include <rt_tm/common/debugging_io.hpp>
#include <rt_tm/common/produce_op.hpp>
#include <rt_tm/common/core_base.hpp>
#include <unordered_set>
#include <variant>
#include <regex>
#include <map>
#include <bit>

namespace rt_tm {

	enum class gguf_metadata_value_type : uint32_t {
		GGUF_METADATA_VALUE_TYPE_UINT8	 = 0,
		GGUF_METADATA_VALUE_TYPE_INT8	 = 1,
		GGUF_METADATA_VALUE_TYPE_UINT16	 = 2,
		GGUF_METADATA_VALUE_TYPE_INT16	 = 3,
		GGUF_METADATA_VALUE_TYPE_UINT32	 = 4,
		GGUF_METADATA_VALUE_TYPE_INT32	 = 5,
		GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
		GGUF_METADATA_VALUE_TYPE_BOOL	 = 7,
		GGUF_METADATA_VALUE_TYPE_STRING	 = 8,
		GGUF_METADATA_VALUE_TYPE_ARRAY	 = 9,
		GGUF_METADATA_VALUE_TYPE_UINT64	 = 10,
		GGUF_METADATA_VALUE_TYPE_INT64	 = 11,
		GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
		GGUF_METADATA_VALUE_TYPE_UNSET	 = 13,
	};

	struct string_iterator {
		const char* first_index{};
		size_t current_index{};
		size_t length{};

		template<typename value_type> RT_TM_FORCE_INLINE bool operator()(size_t size = 0) {
			return (current_index + sizeof(value_type) + size) < length;
		}
	};

	template<typename value_type, auto...> struct value_reader {
		RT_TM_FORCE_INLINE static gguf_metadata_value_type gather_value(string_iterator& input) {
			gguf_metadata_value_type value{};
			if (input.template operator()<value_type>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			if (static_cast<size_t>(value) >= 13) {
				throw std::runtime_error{ "Sorry, but that type is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<uint8_t> {
		RT_TM_FORCE_INLINE static uint8_t gather_value(string_iterator& input) {
			uint8_t value{};
			if (input.template operator()<uint8_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<uint16_t> {
		RT_TM_FORCE_INLINE static uint16_t gather_value(string_iterator& input) {
			uint16_t value{};
			if (input.template operator()<uint16_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<uint32_t> {
		RT_TM_FORCE_INLINE static uint32_t gather_value(string_iterator& input) {
			uint32_t value{};
			if (input.template operator()<uint32_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<uint64_t> {
		RT_TM_FORCE_INLINE static uint64_t gather_value(string_iterator& input) {
			uint64_t value{};
			if (input.template operator()<uint64_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<int8_t> {
		RT_TM_FORCE_INLINE static int8_t gather_value(string_iterator& input) {
			int8_t value{};
			if (input.template operator()<int8_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<int16_t> {
		RT_TM_FORCE_INLINE static int16_t gather_value(string_iterator& input) {
			int16_t value{};
			if (input.template operator()<int16_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<int32_t> {
		RT_TM_FORCE_INLINE static int32_t gather_value(string_iterator& input) {
			int32_t value{};
			if (input.template operator()<int32_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<int64_t> {
		RT_TM_FORCE_INLINE static int64_t gather_value(string_iterator& input) {
			int64_t value{};
			if (input.template operator()<int64_t>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<bool> {
		RT_TM_FORCE_INLINE static bool gather_value(string_iterator& input) {
			bool value{};
			if (input.template operator()<bool>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<float> {
		RT_TM_FORCE_INLINE static float gather_value(string_iterator& input) {
			float value{};
			if (input.template operator()<float>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<double> {
		RT_TM_FORCE_INLINE static double gather_value(string_iterator& input) {
			double value{};
			if (input.template operator()<double>()) {
				std::memcpy(&value, input.first_index + input.current_index, sizeof(value));
				input.current_index += sizeof(value);
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	using gguf_string_t = std::string;

	struct gguf_array_t;

	using gguf_metadata_value_variant = std::variant<float, uint64_t, int64_t, double, bool, gguf_string_t, gguf_array_t*>;

	struct gguf_metadata_value_t {
		gguf_metadata_value_t() noexcept = default;
		gguf_metadata_value_t& operator=(const gguf_metadata_value_t& other) noexcept;
		gguf_metadata_value_t(const gguf_metadata_value_t& other) noexcept {
			*this = other;
		};
		gguf_metadata_value_t(const gguf_metadata_value_variant& other) noexcept;
		gguf_metadata_value_variant value{};
		~gguf_metadata_value_t();
	};

	struct gguf_array_t {
		std::vector<gguf_metadata_value_t> array{};
		gguf_metadata_value_type type{};
	};

	gguf_metadata_value_t::~gguf_metadata_value_t() {
		if (std::holds_alternative<gguf_array_t*>(value)) {
			if (std::get<gguf_array_t*>(value)) {
				delete std::get<gguf_array_t*>(value);
			}
		}
	}

	gguf_metadata_value_t& gguf_metadata_value_t::operator=(const gguf_metadata_value_t& other) noexcept {
		if (std::holds_alternative<float>(other.value)) {
			value.emplace<float>(std::get<float>(other.value));
		} else if (std::holds_alternative<uint64_t>(other.value)) {
			value.emplace<uint64_t>(std::get<uint64_t>(other.value));
		} else if (std::holds_alternative<int64_t>(other.value)) {
			value.emplace<int64_t>(std::get<int64_t>(other.value));
		} else if (std::holds_alternative<double>(other.value)) {
			value.emplace<double>(std::get<double>(other.value));
		} else if (std::holds_alternative<bool>(other.value)) {
			value.emplace<bool>(std::get<bool>(other.value));
		} else if (std::holds_alternative<gguf_string_t>(other.value)) {
			value.emplace<gguf_string_t>(std::get<gguf_string_t>(other.value));
		} else if (std::holds_alternative<gguf_array_t*>(other.value)) {
			if (std::holds_alternative<gguf_array_t*>(value)) {
				if (std::get<gguf_array_t*>(value)) {
					delete std::get<gguf_array_t*>(value);
				}
			}
			value.emplace<gguf_array_t*>(new gguf_array_t{ *std::get<gguf_array_t*>(other.value) });
		}
		return *this;
	};

	gguf_metadata_value_t::gguf_metadata_value_t(const gguf_metadata_value_variant& other) noexcept {
		if (std::holds_alternative<float>(other)) {
			value.emplace<float>(std::get<float>(other));
		} else if (std::holds_alternative<uint64_t>(other)) {
			value.emplace<uint64_t>(std::get<uint64_t>(other));
		} else if (std::holds_alternative<int64_t>(other)) {
			value.emplace<int64_t>(std::get<int64_t>(other));
		} else if (std::holds_alternative<double>(other)) {
			value.emplace<double>(std::get<double>(other));
		} else if (std::holds_alternative<bool>(other)) {
			value.emplace<bool>(std::get<bool>(other));
		} else if (std::holds_alternative<gguf_string_t>(other)) {
			value.emplace<gguf_string_t>(std::get<gguf_string_t>(other));
		} else if (std::holds_alternative<gguf_array_t*>(other)) {
			value.emplace<gguf_array_t*>(new gguf_array_t{ *std::get<gguf_array_t*>(other) });
		}
	};

	template<> struct value_reader<gguf_string_t> {
		RT_TM_FORCE_INLINE static gguf_string_t gather_value(string_iterator& input) {
			uint64_t length{ value_reader<uint64_t>::gather_value(input) };
			constexpr uint64_t MAX_STRING_LENGTH = 1024 * 1024 * 100;
			if (length > MAX_STRING_LENGTH) {
				throw std::runtime_error{ "String length exceeds maximum allowed size!" };
			}
			if (length == 0) {
				return gguf_string_t{};
			}
			gguf_string_t value{};
			value.resize(length);
			if (input.current_index + length < input.length) {
				std::memcpy(value.data(), input.first_index + input.current_index, length);
				input.current_index += length;
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
			return value;
		}
	};

	template<> struct value_reader<gguf_array_t> {
		RT_TM_FORCE_INLINE static gguf_array_t gather_value(string_iterator& input);
	};

	template<> struct value_reader<gguf_metadata_value_variant> {
		RT_TM_INLINE static gguf_metadata_value_variant gather_value(string_iterator& input, gguf_metadata_value_type type) {
			gguf_metadata_value_variant value{};
			switch (type) {
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT8: {
					value.emplace<int64_t>(value_reader<int8_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT16: {
					value.emplace<int64_t>(value_reader<int16_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT32: {
					value.emplace<int64_t>(value_reader<int32_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT64: {
					value.emplace<int64_t>(value_reader<int64_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT8: {
					value.emplace<uint64_t>(value_reader<uint8_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT16: {
					value.emplace<uint64_t>(value_reader<uint16_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32: {
					value.emplace<uint64_t>(value_reader<uint32_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT64: {
					value.emplace<uint64_t>(value_reader<uint64_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_BOOL: {
					value.emplace<bool>(value_reader<bool>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_FLOAT32: {
					value.emplace<float>(value_reader<float>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_FLOAT64: {
					value.emplace<double>(value_reader<double>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING: {
					value.emplace<gguf_string_t>(value_reader<gguf_string_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_ARRAY: {
					value.emplace<gguf_array_t*>(new gguf_array_t{ value_reader<gguf_array_t>::gather_value(input) });
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UNSET: {
					break;
				}
			}
			return value;
		}
	};

	gguf_array_t value_reader<gguf_array_t>::gather_value(string_iterator& input) {
		gguf_metadata_value_type type{ value_reader<gguf_metadata_value_type>::gather_value(input) };
		uint64_t length{ value_reader<uint64_t>::gather_value(input) };
		constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
		if (length > MAX_ARRAY_LENGTH) {
			throw std::runtime_error{ "Array length exceeds maximum allowed size!" };
		}
		gguf_array_t value{};
		value.type = type;
		value.array.reserve(length);
		for (size_t x = 0; x < length; ++x) {
			value.array.emplace_back(value_reader<gguf_metadata_value_variant>::gather_value(input, type));
		}
		return value;
	}

	struct gguf_metadata_kv_t;

	struct gguf_metadata_kv_t {
		gguf_metadata_value_type value_type{};

		gguf_metadata_value_t value{};

		RT_TM_FORCE_INLINE operator bool() const {
			return std::get<bool>(value.value);
		}

		RT_TM_FORCE_INLINE operator int64_t() const {
			return std::get<int64_t>(value.value);
		}

		RT_TM_FORCE_INLINE operator uint64_t() const {
			return std::get<uint64_t>(value.value);
		}

		RT_TM_FORCE_INLINE operator gguf_string_t() const {
			return std::get<gguf_string_t>(value.value);
		}

		RT_TM_FORCE_INLINE operator gguf_array_t() const {
			return *std::get<gguf_array_t*>(value.value);
		}

		RT_TM_FORCE_INLINE operator float() const {
			return std::get<float>(value.value);
		}

		RT_TM_FORCE_INLINE operator double() const {
			return std::get<double>(value.value);
		}
	};

	template<> struct value_reader<gguf_metadata_kv_t> {
		RT_TM_FORCE_INLINE static gguf_metadata_kv_t gather_value(string_iterator& input) {
			gguf_metadata_kv_t value{};
			value.value_type  = value_reader<gguf_metadata_value_type>::gather_value(input);
			value.value.value = value_reader<gguf_metadata_value_variant>::gather_value(input, value.value_type);
			return value;
		}
	};

	struct gguf_header_t {
		std::map<std::string, gguf_metadata_kv_t> metadata_kv{};
		uint64_t metadata_kv_count{};
		uint64_t tensor_count{};
		uint32_t version{};
		uint32_t magic{};
	};

	template<typename value_type> RT_TM_FORCE_INLINE void gather_scalar(const std::string& key, value_type& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<value_type>(v)) {
			out = std::get<value_type>(v);
		}
	};

	template<typename value_type>
	RT_TM_FORCE_INLINE void gather_array(const std::string& key, std::vector<value_type>& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(std::get<value_type>(value.value));
			}
		}
	};

	RT_TM_FORCE_INLINE void print_variant(auto variant) {
		if (std::holds_alternative<float>(variant)) {
			std::cout << "Value: " << std::get<float>(variant) << std::endl;
		} else if (std::holds_alternative<uint64_t>(variant)) {
			std::cout << "Value: " << std::get<uint64_t>(variant) << std::endl;
		} else if (std::holds_alternative<int64_t>(variant)) {
			std::cout << "Value: " << std::get<int64_t>(variant) << std::endl;
		} else if (std::holds_alternative<double>(variant)) {
			std::cout << "Value: " << std::get<double>(variant) << std::endl;
		} else if (std::holds_alternative<bool>(variant)) {
			std::cout << "Value: " << std::get<bool>(variant) << std::endl;
		} else if (std::holds_alternative<gguf_string_t>(variant)) {
			std::cout << "Value: " << std::get<gguf_string_t>(variant) << std::endl;
		} else if (std::holds_alternative<gguf_array_t*>(variant)) {
		}
	}

	template<> struct value_reader<hyper_parameters<model_arch::llama>, model_arch::llama> {
		RT_TM_FORCE_INLINE static hyper_parameters<model_arch::llama> gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
			hyper_parameters<model_arch::llama> value{};
			std::string architecture{};

			if (metadata_kv.contains("general.architecture")) {
				architecture = metadata_kv.at("general.architecture").operator gguf_string_t();
			}

			gather_scalar(architecture + ".block_count", value(llama_hyper_parameter_type_uint64::block_count), metadata_kv);
			gather_scalar(architecture + ".quantization_version", value(llama_hyper_parameter_type_uint64::quantization_version), metadata_kv);
			gather_scalar(architecture + ".rope_dimension_count", value(llama_hyper_parameter_type_uint64::rope_dimension_count), metadata_kv);
			gather_scalar(architecture + ".feed_forward_length", value(llama_hyper_parameter_type_uint64::feed_forward_length), metadata_kv);
			gather_scalar(architecture + ".embedding_length", value(llama_hyper_parameter_type_uint64::embedding_length), metadata_kv);
			gather_scalar(architecture + ".context_length", value(llama_hyper_parameter_type_uint64::context_length), metadata_kv);
			gather_scalar(architecture + ".attention.head_count_kv", value(llama_hyper_parameter_type_uint64::head_count_kv), metadata_kv);
			gather_scalar(architecture + ".attention.head_count", value(llama_hyper_parameter_type_uint64::head_count), metadata_kv);
			gather_scalar(architecture + ".vocab_size", value(llama_hyper_parameter_type_uint64::vocab_size), metadata_kv);
			gather_scalar("general.file_type", value(llama_hyper_parameter_type_uint64::file_type), metadata_kv);
			gather_scalar(architecture + ".attention.layer_norm_rms_epsilon", value(llama_hyper_parameter_type_float64::rms_norm_epsilon), metadata_kv);
			gather_scalar(architecture + ".rope.freq_base", value(llama_hyper_parameter_type_float64::rope_freq_base), metadata_kv);
			gather_scalar("quantize.imatrix.entries_count", value(llama_hyper_parameter_type_uint64::entries_count), metadata_kv);
			gather_scalar("quantize.imatrix.chunks_count", value(llama_hyper_parameter_type_uint64::chunks_count), metadata_kv);

			return value;
		}
	};

	template<> struct value_reader<tokenizer_parameters<model_arch::llama>, model_arch::llama> {
		RT_TM_FORCE_INLINE static tokenizer_parameters<model_arch::llama> gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
			tokenizer_parameters<model_arch::llama> value{};

			gather_scalar("tokenizer.ggml.bos_token_id", value.bos_token_id, metadata_kv);
			gather_scalar("tokenizer.ggml.eos_token_id", value.eos_token_id, metadata_kv);
			gather_scalar("tokenizer.chat_template", value.chat_template, metadata_kv);
			gather_array("tokenizer.ggml.merges", value.merges, metadata_kv);
			gather_scalar("tokenizer.ggml.pre", value.pre, metadata_kv);
			gather_array("tokenizer.ggml.tokens", value.tokens, metadata_kv);
			gather_array("tokenizer.ggml.token_type", value.token_types, metadata_kv);
			return value;
		}
	};

	template<> struct value_reader<gguf_header_t> {
		RT_TM_FORCE_INLINE static gguf_header_t gather_value(string_iterator& input) {
			gguf_header_t value{};
			value.magic = value_reader<uint32_t>::gather_value(input);
			if (value.magic != 0x46554747) {
				throw std::runtime_error{ "Sorry, but that magic value was incorrect!" };
			}
			value.version						  = value_reader<uint32_t>::gather_value(input);
			value.tensor_count					  = value_reader<uint64_t>::gather_value(input);
			value.metadata_kv_count				  = value_reader<uint64_t>::gather_value(input);
			constexpr uint64_t MAX_TENSOR_COUNT	  = 100000;
			constexpr uint64_t MAX_METADATA_COUNT = 10000;
			if (value.tensor_count > MAX_TENSOR_COUNT) {
				throw std::runtime_error{ "Tensor count exceeds reasonable maximum!" };
			}
			if (value.metadata_kv_count > MAX_METADATA_COUNT) {
				throw std::runtime_error{ "Metadata count exceeds reasonable maximum!" };
			}
			for (size_t x = 0; x < value.metadata_kv_count; ++x) {
				std::string new_string		  = value_reader<gguf_string_t>::gather_value(input);
				value.metadata_kv[new_string] = value_reader<gguf_metadata_kv_t>::gather_value(input);
			}
			return value;
		}
	};

	struct gguf_tensor_info_t {
		std::vector<uint64_t> dimensions{};
		uint32_t n_dimensions{};
		std::string name{};
		uint64_t offset{};
		data_type type{};
	};

	template<model_arch arch> struct string_to_tensor_name;

	template<> struct string_to_tensor_name<model_arch::llama> {
		RT_TM_FORCE_INLINE static size_t impl(std::string_view input) noexcept {
			if (input == "token_embd.weight")
				return static_cast<size_t>(llama_op_names::token_embed);
			if (input == "rope_freqs.weight")
				return static_cast<size_t>(llama_op_names::rope_freqs);
			if (input == "output_norm.weight")
				return static_cast<size_t>(llama_op_names::output_norm);
			if (input == "output.weight")
				return static_cast<size_t>(llama_op_names::blk_output);

			if (input.find(".attn_q.weight") != std::string_view::npos)
				return static_cast<size_t>(llama_op_names::blk_attn_q);
			if (input.find(".attn_norm.weight") != std::string_view::npos)
				return static_cast<size_t>(llama_op_names::blk_attn_norm);

			if (input.starts_with("blk.") && input.ends_with(".weight")) {
				auto second_dot = input.find('.', 4);
				if (second_dot != std::string_view::npos) {
					auto suffix = input.substr(second_dot + 1);

					if (suffix == "attn_q.weight")
						return static_cast<size_t>(llama_op_names::blk_attn_q);
					if (suffix == "attn_norm.weight")
						return static_cast<size_t>(llama_op_names::blk_attn_norm);
					if (suffix == "attn_k.weight")
						return static_cast<size_t>(llama_op_names::blk_attn_k);
					if (suffix == "attn_v.weight")
						return static_cast<size_t>(llama_op_names::blk_attn_v);
					if (suffix == "ffn_down.weight")
						return static_cast<size_t>(llama_op_names::blk_ffn_down);
					if (suffix == "ffn_gate.weight")
						return static_cast<size_t>(llama_op_names::blk_ffn_gate);
					if (suffix == "attn_output.weight")
						return static_cast<size_t>(llama_op_names::blk_attn_output);
					if (suffix == "ffn_norm.weight")
						return static_cast<size_t>(llama_op_names::blk_ffn_norm);
					if (suffix == "ffn_up.weight")
						return static_cast<size_t>(llama_op_names::blk_ffn_up);
				}
			}

			return static_cast<size_t>(llama_op_names::count);
		}
	};

	template<> struct value_reader<gguf_tensor_info_t> {
		RT_TM_FORCE_INLINE static gguf_tensor_info_t gather_value(string_iterator& input) {
			gguf_tensor_info_t value{};
			value.name						  = value_reader<gguf_string_t>::gather_value(input);
			value.n_dimensions				  = value_reader<uint32_t>::gather_value(input);
			constexpr uint32_t MAX_DIMENSIONS = 8;
			if (value.n_dimensions > MAX_DIMENSIONS) {
				throw std::runtime_error{ "Tensor dimensions exceed maximum!" };
			}
			for (size_t x = 0; x < value.n_dimensions; ++x) {
				uint64_t dim					= value_reader<uint64_t>::gather_value(input);
				constexpr uint64_t MAX_DIM_SIZE = 1ULL << 32;
				if (dim > MAX_DIM_SIZE) {
					throw std::runtime_error{ "Tensor dimension size too large!" };
				}
				value.dimensions.emplace_back(dim);
			}
			value.type	 = static_cast<data_type>(value_reader<uint32_t>::gather_value(input));
			value.offset = value_reader<uint64_t>::gather_value(input);
			return value;
		}
	};

	RT_TM_FORCE_INLINE bool operator<(const gguf_tensor_info_t& lhs, const gguf_tensor_info_t& rhs) noexcept {
		const std::string& lhs_name{ lhs.name };
		const std::string& rhs_name{ rhs.name };
		if (lhs_name.find_first_of("1234567890") != std::string::npos && rhs_name.find_first_of("1234567890") != std::string::npos) {
			size_t lhs_offset{ lhs_name.find_first_of("1234567890") };
			size_t rhs_offset{ rhs_name.find_first_of("1234567890") };
			std::string lhs_val_raw{ lhs_name.substr(lhs_offset, lhs_name.find_first_not_of("1234567890", lhs_offset) - lhs_offset + 1) };
			std::string rhs_val_raw{ rhs_name.substr(rhs_offset, rhs_name.find_first_not_of("1234567890", rhs_offset) - rhs_offset + 1) };
			auto* lhs_ptr_01 = lhs_val_raw.data();
			auto* lhs_ptr_02 = lhs_val_raw.data() + lhs_val_raw.size();
			size_t lhs_val{ std::strtoull(lhs_ptr_01, &lhs_ptr_02, 10) };
			auto* rhs_ptr_01 = rhs_val_raw.data();
			auto* rhs_ptr_02 = rhs_val_raw.data() + rhs_val_raw.size();
			size_t rhs_val{ std::strtoull(rhs_ptr_01, &rhs_ptr_02, 10) };
			return lhs_val < rhs_val;
		} else {
			return lhs_name < rhs_name;
		}
	}

	RT_TM_FORCE_INLINE void sort_tensor_infos(std::vector<gguf_tensor_info_t>& tensor_infos) noexcept {
		std::sort(tensor_infos.begin(), tensor_infos.end(), std::less<gguf_tensor_info_t>{});
	}

	RT_TM_FORCE_INLINE constexpr size_t parse_number(std::string_view str) noexcept {
		size_t result = 0;
		for (char c: str) {
			if (c >= '0' && c <= '9') {
				result = result * 10 + (c - '0');
			} else {
				break;
			}
		}
		return result;
	}

	RT_TM_FORCE_INLINE constexpr size_t extract_layer_number(std::string_view name) noexcept {
		if RT_TM_LIKELY (name[0] == 'c' && name.starts_with("cache_")) {
			for (size_t i = 7; i < name.size(); ++i) {
				if (name[i] == 'l' && i + 1 < name.size()) {
					return parse_number(name.substr(i + 1));
				}
			}
			return 0;
		}
		if RT_TM_LIKELY (name[0] == 'b' && name.starts_with("blk.")) {
			size_t start = 4;
			size_t end	 = name.find('.', start);
			if (end != std::string_view::npos) {
				return parse_number(name.substr(start, end - start));
			}
		}

		return 0;
	}

	struct gguf_file_t {
		std::vector<gguf_tensor_info_t> tensor_infos{};
		std::vector<uint8_t> tensor_data{};
		std::vector<uint8_t> _padding{};
		gguf_header_t header{};
	};

	uint64_t align_offset(uint64_t offset, uint64_t alignment = 1) {
		alignment = alignment == 0 ? 1 : alignment;
		return offset + (alignment - (offset % alignment)) % alignment;
	}

	enum class model_format { gguf = 1 };

	template<global_config config, model_arch arch, model_format type> struct model_parser;

	template<global_config config> struct model_parser<config, model_arch::llama, model_format::gguf> {
		static_assert((std::endian::native == std::endian::little), "Sorry, but big-endian is not yet supported by the library");

		RT_TM_FORCE_INLINE static void generate_ops(model_graph<config>& model) {
			model.op_cores.reserve(3000);

			const auto& hparams				  = model.hparams;
			const size_t block_count		  = hparams(llama_hyper_parameter_type_uint64::block_count);
			const size_t vocab_size			  = hparams(llama_hyper_parameter_type_uint64::vocab_size);
			const size_t embedding_length	  = hparams(llama_hyper_parameter_type_uint64::embedding_length);
			const size_t head_count			  = hparams(llama_hyper_parameter_type_uint64::head_count);
			const size_t head_count_kv		  = hparams(llama_hyper_parameter_type_uint64::head_count_kv);
			const size_t feed_forward_length  = hparams(llama_hyper_parameter_type_uint64::feed_forward_length);
			const size_t context_length		  = hparams(llama_hyper_parameter_type_uint64::context_length);
			const size_t rope_dimension_count = hparams(llama_hyper_parameter_type_uint64::rope_dimension_count);
			const double rms_norm_epsilon	  = hparams(llama_hyper_parameter_type_float64::rms_norm_epsilon);
			const double rope_freq_base		  = hparams(llama_hyper_parameter_type_float64::rope_freq_base);

			const size_t head_dim							= embedding_length / head_count;
			const size_t effective_context_length_per_layer = context_length / block_count;
			const size_t combined_kv_head_dim				= head_dim * head_count_kv;

			size_t op_id_counter = model.op_cores.size();

			for (size_t x = 0; x < block_count; ++x) {
				core_base_creation_data k_cache{};
				k_cache.allocated_dims = { combined_kv_head_dim, effective_context_length_per_layer, 1, 1 };
				k_cache.allocated_dims = { combined_kv_head_dim, effective_context_length_per_layer, 1, 1 };
				k_cache.data_type_val  = data_type::float_16;
				k_cache.op_id		   = op_id_counter++;
				k_cache.type		   = op_type::noop;
				k_cache.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::cache_k)][x];
				model.op_cores.emplace_back(k_cache);
				core_base_creation_data v_cache{};
				v_cache.allocated_dims = { combined_kv_head_dim, effective_context_length_per_layer, 1, 1 };
				v_cache.allocated_dims = { combined_kv_head_dim, effective_context_length_per_layer, 1, 1 };
				v_cache.data_type_val  = data_type::float_16;
				v_cache.op_id		   = op_id_counter++;
				v_cache.type		   = op_type::noop;
				v_cache.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::cache_v)][x];
				model.op_cores.emplace_back(v_cache);
			}

			std::unordered_map<std::string, core_base_creation_data*> weight_map;

			auto copy_rope_params = [&](core_base_creation_data& op) {
				op.aux_params.resize(static_cast<size_t>(rope_aux_params::count));
				std::memcpy(op.aux_params.data(), &rope_dimension_count, sizeof(rope_dimension_count));
				std::memcpy(op.aux_params.data() + sizeof(rope_dimension_count), &rope_freq_base, sizeof(rope_freq_base));
			};

			auto copy_norm_params = [&](core_base_creation_data& op) {
				op.aux_params.resize(static_cast<size_t>(rms_norm_aux_params::count));
				std::memcpy(op.aux_params.data(), &rms_norm_epsilon, sizeof(rms_norm_epsilon));
			};

			auto copy_attn_params = [&](core_base_creation_data& op) {
				op.aux_params.resize(static_cast<size_t>(attention_aux_params::count));
				std::memcpy(op.aux_params.data(), &rms_norm_epsilon, sizeof(rms_norm_epsilon));
			};

			auto copy_ffn_params = [&](core_base_creation_data& op) {
				op.aux_params.resize(static_cast<size_t>(ffn_aux_params::count));
				std::memcpy(op.aux_params.data(), &rms_norm_epsilon, sizeof(rms_norm_epsilon));
			};

			core_base_creation_data rope_dims{};
			rope_dims.type			 = op_type::noop;
			rope_dims.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::rope_dims)][0];
			rope_dims.allocated_dims = { 2, 1, 1, 1 };
			rope_dims.allocated_dims = { 2, 1, 1, 1 };
			rope_dims.data_type_val	 = data_type::int_32;
			rope_dims.depth			 = 0;
			rope_dims.op_id			 = op_id_counter++;
			model.op_cores.emplace_back(rope_dims);

			for (size_t x = 0; x < model.op_cores.size(); ++x) {
				weight_map[model.op_cores[x].name] = &model.op_cores[x];
			}

			core_base_creation_data input_tokens{};
			input_tokens.type			= op_type::noop;
			input_tokens.name			= model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::input_tokens)][0];
			input_tokens.allocated_dims = { 2, 1, 1, 1 };
			input_tokens.allocated_dims = { 2, 1, 1, 1 };
			input_tokens.data_type_val	= data_type::int_32;
			input_tokens.depth			= 0;
			input_tokens.op_id			= op_id_counter++;
			model.op_cores.emplace_back(input_tokens);

			core_base_creation_data inp_embd_op{ produce_op<model_arch_traits<model_arch::llama>, op_type::get_rows>::impl(*weight_map.at("token_embd.weight"), input_tokens,
				op_id_counter++, llama_op_names::input_embedding, 0) };
			model.op_cores.emplace_back(inp_embd_op);
			core_base_creation_data* current_input = &model.op_cores.back();

			for (size_t block_idXd = 0; block_idXd < block_count; ++block_idXd) {
				core_base_creation_data inp_embd_op{ produce_op<model_arch_traits<model_arch::llama>, op_type::rms_norm>::impl(*current_input, op_id_counter++,
					llama_op_names::norm, block_idXd, rms_norm_epsilon) };
				model.op_cores.emplace_back(inp_embd_op);

				core_base_creation_data attn_norm_op{ produce_op<model_arch_traits<model_arch::llama>, op_type::mul>::impl(inp_embd_op,
					*weight_map.at("blk." + std::to_string(block_idXd) + ".attn_norm.weight"), op_id_counter++, llama_op_names::attn_norm, block_idXd) };
				model.op_cores.emplace_back(std::move(attn_norm_op));

				core_base_creation_data q_proj{ produce_op<model_arch_traits<model_arch::llama>, op_type::mul_mat>::impl(attn_norm_op,
					*weight_map.at("blk." + std::to_string(block_idXd) + ".attn_q.weight"), op_id_counter++, llama_op_names::q_proj, block_idXd) };
				model.op_cores.emplace_back(std::move(q_proj));

				core_base_creation_data q_reshape{ produce_op<model_arch_traits<model_arch::llama>, op_type::reshape>::impl(q_proj, op_id_counter++, llama_op_names::q_reshape,
					block_idXd, head_dim, head_count, 2) };
				model.op_cores.emplace_back(std::move(q_reshape));

				core_base_creation_data rope_q{ produce_op<model_arch_traits<model_arch::llama>, op_type::mul_mat>::impl(
					*weight_map.at("blk." + std::to_string(block_idXd) + ".attn_q.weight"), attn_norm_op, op_id_counter++, llama_op_names::rope_q, block_idXd) };
				model.op_cores.emplace_back(std::move(rope_q));

				//core_base_creation_data* q_proj_out = &model.op_cores.back();
				/*
				core_base_creation_data qcur_op{};
				qcur_op.type		  = op_type::mul_mat;
				qcur_op.op_id		  = op_id_counter++;
				qcur_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::query_current)][block_idXd];
				qcur_op.allocated_dims		  = { embedding_length, 2, 1, 1 };
				qcur_op.data_type_val = attn_norm_out->data_type_val;
				qcur_op.input_ops	  = { { attn_norm_out, weight_map.at("blk." + std::to_string(block_idXd) + ".attn_q.weight") } };
				qcur_op.depth		  = std::max(attn_norm_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".attn_q.weight")->depth) + 1;
				attn_norm_out->dependent_ops.emplace_back(qcur_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".attn_q.weight")->dependent_ops.emplace_back(qcur_op.op_id);
				copy_attn_params(qcur_op);
				model.op_cores.emplace_back(std::move(qcur_op));
				core_base_creation_data* qcur_out = &model.op_cores.back();

				core_base_creation_data kcur_op{};
				kcur_op.type		  = op_type::mul_mat;
				kcur_op.op_id		  = op_id_counter++;
				kcur_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_current)][block_idXd];
				kcur_op.allocated_dims		  = { head_count_kv * head_dim, 2, 1, 1 };
				kcur_op.data_type_val = attn_norm_out->data_type_val;
				kcur_op.input_ops	  = { { attn_norm_out, weight_map.at("blk." + std::to_string(block_idXd) + ".attn_k.weight") } };
				kcur_op.depth		  = std::max(attn_norm_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".attn_k.weight")->depth) + 1;
				attn_norm_out->dependent_ops.emplace_back(kcur_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".attn_k.weight")->dependent_ops.emplace_back(kcur_op.op_id);
				copy_attn_params(kcur_op);
				model.op_cores.emplace_back(std::move(kcur_op));
				core_base_creation_data* kcur_out = &model.op_cores.back();

				core_base_creation_data vcur_op{};
				vcur_op.type		  = op_type::mul_mat;
				vcur_op.op_id		  = op_id_counter++;
				vcur_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_current)][block_idXd];
				vcur_op.allocated_dims		  = { head_count_kv * head_dim, 2, 1, 1 };
				vcur_op.data_type_val = attn_norm_out->data_type_val;
				vcur_op.input_ops	  = { { attn_norm_out,  weight_map.at("blk." + std::to_string(block_idXd) + ".attn_v.weight") } };
				vcur_op.depth		  = std::max(attn_norm_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".attn_v.weight")->depth) + 1;
				attn_norm_out->dependent_ops.emplace_back(vcur_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".attn_v.weight")->dependent_ops.emplace_back(vcur_op.op_id);
				copy_attn_params(vcur_op);
				model.op_cores.emplace_back(std::move(vcur_op));
				core_base_creation_data* vcur_out = &model.op_cores.back();

				core_base_creation_data qcur_reshaped_op{};
				qcur_reshaped_op.type		   = op_type::reshape;
				qcur_reshaped_op.op_id		   = op_id_counter++;
				qcur_reshaped_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::query_reshaped)][block_idXd];
				qcur_reshaped_op.allocated_dims		   = { head_dim, head_count, 2, 1 };
				qcur_reshaped_op.data_type_val = qcur_out->data_type_val;
				qcur_reshaped_op.input_ops	   = { { qcur_out } };
				qcur_reshaped_op.depth		   = qcur_out->depth + 1;
				qcur_out->dependent_ops.emplace_back(qcur_reshaped_op.op_id);
				copy_attn_params(qcur_reshaped_op);
				model.op_cores.emplace_back(std::move(qcur_reshaped_op));
				core_base_creation_data* qcur_reshaped = &model.op_cores.back();

				core_base_creation_data kcur_reshaped_op{};
				kcur_reshaped_op.type		   = op_type::reshape;
				kcur_reshaped_op.op_id		   = op_id_counter++;
				kcur_reshaped_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_reshaped)][block_idXd];
				kcur_reshaped_op.allocated_dims		   = { head_dim, head_count_kv, 2, 1 };
				kcur_reshaped_op.data_type_val = kcur_out->data_type_val;
				kcur_reshaped_op.input_ops	   = { { kcur_out } };
				kcur_reshaped_op.depth		   = kcur_out->depth + 1;
				kcur_out->dependent_ops.emplace_back(kcur_reshaped_op.op_id);
				copy_attn_params(kcur_reshaped_op);
				model.op_cores.emplace_back(std::move(kcur_reshaped_op));
				core_base_creation_data* kcur_reshaped = &model.op_cores.back();

				core_base_creation_data vcur_reshaped_op{};
				vcur_reshaped_op.type		   = op_type::reshape;
				vcur_reshaped_op.op_id		   = op_id_counter++;
				vcur_reshaped_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_reshaped)][block_idXd];
				vcur_reshaped_op.allocated_dims		   = { head_count_kv * head_dim, 2, 1, 1 };
				vcur_reshaped_op.data_type_val = vcur_out->data_type_val;
				vcur_reshaped_op.input_ops	   = { { vcur_out } };
				vcur_reshaped_op.depth		   = vcur_out->depth + 1;
				vcur_out->dependent_ops.emplace_back(vcur_reshaped_op.op_id);
				copy_attn_params(vcur_reshaped_op);
				model.op_cores.emplace_back(std::move(vcur_reshaped_op));
				core_base_creation_data* vcur_reshaped = &model.op_cores.back();

				core_base_creation_data qcur_rope_op{};
				qcur_rope_op.type		   = op_type::rope;
				qcur_rope_op.op_id		   = op_id_counter++;
				qcur_rope_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::query_rope)][block_idXd];
				qcur_rope_op.allocated_dims		   = qcur_reshaped->allocated_dims;
				qcur_rope_op.data_type_val = qcur_reshaped->data_type_val;
				qcur_rope_op.input_ops	   = { { qcur_reshaped, weight_map.at("rope_dims.weight"), weight_map.at("rope_freqs.weight") } };
				qcur_rope_op.depth		   = std::max(qcur_reshaped->depth, weight_map.at("rope_freqs.weight")->depth) + 1;
				qcur_reshaped->dependent_ops.emplace_back(qcur_rope_op.op_id);
				weight_map.at("rope_freqs.weight")->dependent_ops.emplace_back(qcur_rope_op.op_id);
				copy_rope_params(qcur_rope_op);
				model.op_cores.emplace_back(std::move(qcur_rope_op));
				core_base_creation_data* qcur_rope = &model.op_cores.back();

				core_base_creation_data kcur_rope_op{};
				kcur_rope_op.type		   = op_type::rope;
				kcur_rope_op.op_id		   = op_id_counter++;
				kcur_rope_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_rope)][block_idXd];
				kcur_rope_op.allocated_dims		   = kcur_reshaped->allocated_dims;
				kcur_rope_op.data_type_val = kcur_reshaped->data_type_val;
				kcur_rope_op.input_ops	   = { { kcur_reshaped,  weight_map.at("rope_freqs.weight") } };
				kcur_rope_op.depth		   = std::max(kcur_reshaped->depth, weight_map.at("rope_freqs.weight")->depth) + 1;
				kcur_reshaped->dependent_ops.emplace_back(kcur_rope_op.op_id);
				weight_map.at("rope_freqs.weight")->dependent_ops.emplace_back(kcur_rope_op.op_id);
				copy_rope_params(kcur_rope_op);
				model.op_cores.emplace_back(std::move(kcur_rope_op));
				core_base_creation_data* kcur_rope = &model.op_cores.back();

				core_base_creation_data cache_k_view_op{};
				cache_k_view_op.type		  = op_type::view;
				cache_k_view_op.op_id		  = op_id_counter++;
				cache_k_view_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_cache_view)][block_idXd];
				cache_k_view_op.allocated_dims		  = { context_length, 1, 1, 1 };
				cache_k_view_op.data_type_val = data_type::float_16;
				cache_k_view_op.input_ops	  = { weight_map.at("cache_k_l" + std::to_string(block_idXd)) };
				cache_k_view_op.depth		  = weight_map.at("cache_k_l" + std::to_string(block_idXd))->depth + 1;
				weight_map.at("cache_k_l" + std::to_string(block_idXd))->dependent_ops.emplace_back(cache_k_view_op.op_id);
				model.op_cores.emplace_back(std::move(cache_k_view_op));
				core_base_creation_data* cache_k_view = &model.op_cores.back();

				core_base_creation_data cache_k_copy_op{};
				cache_k_copy_op.type		  = op_type::copy;
				cache_k_copy_op.op_id		  = op_id_counter++;
				cache_k_copy_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_cache_copy)][block_idXd];
				cache_k_copy_op.allocated_dims		  = kcur_rope->allocated_dims;
				cache_k_copy_op.data_type_val = kcur_rope->data_type_val;
				cache_k_copy_op.input_ops	  = { kcur_rope };
				cache_k_copy_op.depth		  = kcur_rope->depth + 1;
				kcur_rope->dependent_ops.emplace_back(cache_k_copy_op.op_id);
				model.op_cores.emplace_back(std::move(cache_k_copy_op));

				core_base_creation_data vcur_reshaped2_op{};
				vcur_reshaped2_op.type			= op_type::reshape;
				vcur_reshaped2_op.op_id			= op_id_counter++;
				vcur_reshaped2_op.name			= model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_reshaped)][block_idXd];
				vcur_reshaped2_op.allocated_dims			= { head_count_kv * head_dim, 2, 1, 1 };
				vcur_reshaped2_op.data_type_val = vcur_reshaped->data_type_val;
				vcur_reshaped2_op.input_ops		= { vcur_reshaped };
				vcur_reshaped2_op.depth			= vcur_reshaped->depth + 1;
				vcur_reshaped->dependent_ops.emplace_back(vcur_reshaped2_op.op_id);
				copy_attn_params(vcur_reshaped2_op);
				model.op_cores.emplace_back(std::move(vcur_reshaped2_op));
				core_base_creation_data* vcur_reshaped2 = &model.op_cores.back();

				core_base_creation_data vcur_transposed_op{};
				vcur_transposed_op.type			 = op_type::transpose;
				vcur_transposed_op.op_id		 = op_id_counter++;
				vcur_transposed_op.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_transposed)][block_idXd];
				vcur_transposed_op.allocated_dims			 = { 2, head_count_kv * head_dim, 1, 1 };
				vcur_transposed_op.data_type_val = vcur_reshaped2->data_type_val;
				vcur_transposed_op.input_ops	 = { vcur_reshaped2 };
				vcur_transposed_op.depth		 = vcur_reshaped2->depth + 1;
				vcur_reshaped2->dependent_ops.emplace_back(vcur_transposed_op.op_id);
				model.op_cores.emplace_back(std::move(vcur_transposed_op));
				core_base_creation_data* vcur_transposed = &model.op_cores.back();

				core_base_creation_data cache_v_view_op{};
				cache_v_view_op.type		  = op_type::view;
				cache_v_view_op.op_id		  = op_id_counter++;
				cache_v_view_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_cache_view)][block_idXd];
				cache_v_view_op.allocated_dims		  = { 2, head_count_kv * head_dim, 1, 1 };
				cache_v_view_op.data_type_val = data_type::float_16;
				cache_v_view_op.input_ops	  = { weight_map.at("cache_v_l" + std::to_string(block_idXd)) };
				cache_v_view_op.depth		  = weight_map.at("cache_v_l" + std::to_string(block_idXd))->depth + 1;
				weight_map.at("cache_v_l" + std::to_string(block_idXd))->dependent_ops.emplace_back(cache_v_view_op.op_id);
				model.op_cores.emplace_back(std::move(cache_v_view_op));
				core_base_creation_data* cache_v_view = &model.op_cores.back();

				core_base_creation_data cache_v_copy_op{};
				cache_v_copy_op.type		  = op_type::copy;
				cache_v_copy_op.op_id		  = op_id_counter++;
				cache_v_copy_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_cache_copy)][block_idXd];
				cache_v_copy_op.allocated_dims		  = vcur_transposed->allocated_dims;
				cache_v_copy_op.data_type_val = vcur_transposed->data_type_val;
				cache_v_copy_op.input_ops	  = { vcur_transposed };
				cache_v_copy_op.depth		  = vcur_transposed->depth + 1;
				vcur_transposed->dependent_ops.emplace_back(cache_v_copy_op.op_id);
				model.op_cores.emplace_back(std::move(cache_v_copy_op));

				core_base_creation_data cache_v_view2_op{};
				cache_v_view2_op.type		   = op_type::view;
				cache_v_view2_op.op_id		   = op_id_counter++;
				cache_v_view2_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_cache_view)][block_idXd];
				cache_v_view2_op.allocated_dims		   = { 2, head_count_kv * head_dim, 1, 1 };
				cache_v_view2_op.data_type_val = cache_v_view->data_type_val;
				cache_v_view2_op.input_ops	   = { cache_v_view };
				cache_v_view2_op.depth		   = cache_v_view->depth + 1;
				cache_v_view->dependent_ops.emplace_back(cache_v_view2_op.op_id);
				model.op_cores.emplace_back(std::move(cache_v_view2_op));
				core_base_creation_data* cache_v_view2 = &model.op_cores.back();

				core_base_creation_data cache_v_permuted_op{};
				cache_v_permuted_op.type		  = op_type::permute;
				cache_v_permuted_op.op_id		  = op_id_counter++;
				cache_v_permuted_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::value_cache_permuted)][block_idXd];
				cache_v_permuted_op.allocated_dims		  = { 1, 1, head_count_kv, head_dim };
				cache_v_permuted_op.data_type_val = cache_v_view2->data_type_val;
				cache_v_permuted_op.input_ops	  = { cache_v_view2 };
				cache_v_permuted_op.depth		  = cache_v_view2->depth + 1;
				cache_v_view2->dependent_ops.emplace_back(cache_v_permuted_op.op_id);
				model.op_cores.emplace_back(std::move(cache_v_permuted_op));
				core_base_creation_data* cache_v_permuted = &model.op_cores.back();

				core_base_creation_data cache_k_view2_op{};
				cache_k_view2_op.type		   = op_type::view;
				cache_k_view2_op.op_id		   = op_id_counter++;
				cache_k_view2_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_cache_view)][block_idXd];
				cache_k_view2_op.allocated_dims		   = { 1, 1, head_count_kv, head_dim };
				cache_k_view2_op.data_type_val = cache_k_view->data_type_val;
				cache_k_view2_op.input_ops	   = { cache_k_view };
				cache_k_view2_op.depth		   = cache_k_view->depth + 1;
				cache_k_view->dependent_ops.emplace_back(cache_k_view2_op.op_id);
				model.op_cores.emplace_back(std::move(cache_k_view2_op));
				core_base_creation_data* cache_k_view2 = &model.op_cores.back();

				core_base_creation_data cache_k_permuted_op{};
				cache_k_permuted_op.type		  = op_type::permute;
				cache_k_permuted_op.op_id		  = op_id_counter++;
				cache_k_permuted_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::key_cache_permuted)][block_idXd];
				cache_k_permuted_op.allocated_dims		  = { head_dim, head_count, head_count_kv, 1 };
				cache_k_permuted_op.data_type_val = cache_k_view2->data_type_val;
				cache_k_permuted_op.input_ops	  = { cache_k_view2 };
				cache_k_permuted_op.depth		  = cache_k_view2->depth + 1;
				cache_k_view2->dependent_ops.emplace_back(cache_k_permuted_op.op_id);
				model.op_cores.emplace_back(std::move(cache_k_permuted_op));
				core_base_creation_data* cache_k_permuted = &model.op_cores.back();

				core_base_creation_data qcur_permuted_op{};
				qcur_permuted_op.type		   = op_type::permute;
				qcur_permuted_op.op_id		   = op_id_counter++;
				qcur_permuted_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::query_permuted)][block_idXd];
				qcur_permuted_op.allocated_dims		   = { 1, head_count, 1, head_dim };
				qcur_permuted_op.data_type_val = qcur_rope->data_type_val;
				qcur_permuted_op.input_ops	   = { qcur_rope };
				qcur_permuted_op.depth		   = qcur_rope->depth + 1;
				qcur_rope->dependent_ops.emplace_back(qcur_permuted_op.op_id);
				copy_attn_params(qcur_permuted_op);
				model.op_cores.emplace_back(std::move(qcur_permuted_op));
				core_base_creation_data* qcur_permuted = &model.op_cores.back();

				core_base_creation_data attn_scores_op{};
				attn_scores_op.type			 = op_type::mul_mat;
				attn_scores_op.op_id		 = op_id_counter++;
				attn_scores_op.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::attention_scores)][block_idXd];
				attn_scores_op.allocated_dims			 = { 1, head_count, 1, 1 };
				attn_scores_op.data_type_val = qcur_permuted->data_type_val;
				attn_scores_op.input_ops	 = { qcur_permuted,  cache_k_permuted };
				attn_scores_op.depth		 = std::max(qcur_permuted->depth, cache_k_permuted->depth) + 1;
				qcur_permuted->dependent_ops.emplace_back(attn_scores_op.op_id);
				cache_k_permuted->dependent_ops.emplace_back(attn_scores_op.op_id);
				copy_attn_params(attn_scores_op);
				model.op_cores.emplace_back(std::move(attn_scores_op));
				core_base_creation_data* attn_scores = &model.op_cores.back();

				core_base_creation_data softmaXd_op{};
				softmaXd_op.type		  = op_type::softmax;
				softmaXd_op.op_id		  = op_id_counter++;
				softmaXd_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::attention_weights)][block_idXd];
				softmaXd_op.allocated_dims		  = attn_scores->allocated_dims;
				softmaXd_op.data_type_val = attn_scores->data_type_val;
				softmaXd_op.input_ops	  = { attn_scores };
				softmaXd_op.depth		  = attn_scores->depth + 1;
				attn_scores->dependent_ops.emplace_back(softmaXd_op.op_id);
				model.op_cores.emplace_back(std::move(softmaXd_op));
				core_base_creation_data* softmaXd_out = &model.op_cores.back();

				core_base_creation_data attn_weighted_op{};
				attn_weighted_op.type		   = op_type::mul_mat;
				attn_weighted_op.op_id		   = op_id_counter++;
				attn_weighted_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::attention_output)][block_idXd];
				attn_weighted_op.allocated_dims		   = { 1, head_count, 1, head_dim };
				attn_weighted_op.data_type_val = softmaXd_out->data_type_val;
				attn_weighted_op.input_ops	   = { softmaXd_out,  cache_v_permuted };
				attn_weighted_op.depth		   = std::max(softmaXd_out->depth, cache_v_permuted->depth) + 1;
				softmaXd_out->dependent_ops.emplace_back(attn_weighted_op.op_id);
				cache_v_permuted->dependent_ops.emplace_back(attn_weighted_op.op_id);
				copy_attn_params(attn_weighted_op);
				model.op_cores.emplace_back(std::move(attn_weighted_op));
				core_base_creation_data* attn_weighted = &model.op_cores.back();

				core_base_creation_data attn_permuted_op{};
				attn_permuted_op.type		   = op_type::permute;
				attn_permuted_op.op_id		   = op_id_counter++;
				attn_permuted_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::attention_result)][0];
				attn_permuted_op.allocated_dims		   = { 1, 1, head_count, head_dim };
				attn_permuted_op.data_type_val = attn_weighted->data_type_val;
				attn_permuted_op.input_ops	   = { attn_weighted };
				attn_permuted_op.depth		   = attn_weighted->depth + 1;
				attn_weighted->dependent_ops.emplace_back(attn_permuted_op.op_id);
				model.op_cores.emplace_back(std::move(attn_permuted_op));
				core_base_creation_data* attn_permuted = &model.op_cores.back();

				core_base_creation_data kqv_out_op{};
				kqv_out_op.type			 = op_type::cont;
				kqv_out_op.op_id		 = op_id_counter++;
				kqv_out_op.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::attention_final)][block_idXd];
				kqv_out_op.allocated_dims			 = { embedding_length, 2, 1, 1 };
				kqv_out_op.data_type_val = attn_permuted->data_type_val;
				kqv_out_op.input_ops	 = { attn_permuted };
				kqv_out_op.depth		 = attn_permuted->depth + 1;
				attn_permuted->dependent_ops.emplace_back(kqv_out_op.op_id);
				model.op_cores.emplace_back(std::move(kqv_out_op));
				core_base_creation_data* kqv_out = &model.op_cores.back();

				core_base_creation_data attn_out_op{};
				attn_out_op.type		  = op_type::mul_mat;
				attn_out_op.op_id		  = op_id_counter++;
				attn_out_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::attention_projection)][block_idXd];
				attn_out_op.allocated_dims		  = { embedding_length, 2, 1, 1 };
				attn_out_op.data_type_val = kqv_out->data_type_val;
				attn_out_op.input_ops	  = { kqv_out,  weight_map.at("blk." + std::to_string(block_idXd) + ".attn_output.weight") };
				attn_out_op.depth		  = std::max(kqv_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".attn_output.weight")->depth) + 1;
				kqv_out->dependent_ops.emplace_back(attn_out_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".attn_output.weight")->dependent_ops.emplace_back(attn_out_op.op_id);
				copy_attn_params(attn_out_op);
				model.op_cores.emplace_back(std::move(attn_out_op));
				core_base_creation_data* attn_out = &model.op_cores.back();

				core_base_creation_data ffn_inp_op{};
				ffn_inp_op.type			 = op_type::add;
				ffn_inp_op.op_id		 = op_id_counter++;
				ffn_inp_op.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_input)][block_idXd];
				ffn_inp_op.allocated_dims			 = current_input->allocated_dims;
				ffn_inp_op.data_type_val = current_input->data_type_val;
				ffn_inp_op.input_ops	 = { current_input,  attn_out };
				ffn_inp_op.depth		 = attn_out->depth + 1;
				current_input->dependent_ops.emplace_back(ffn_inp_op.op_id);
				attn_out->dependent_ops.emplace_back(ffn_inp_op.op_id);
				model.op_cores.emplace_back(std::move(ffn_inp_op));
				core_base_creation_data* ffn_inp = &model.op_cores.back();

				core_base_creation_data ffn_norm_rms_op{};
				ffn_norm_rms_op.type		  = op_type::rms_norm;
				ffn_norm_rms_op.op_id		  = op_id_counter++;
				ffn_norm_rms_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::layer_norm)][block_idXd];
				ffn_norm_rms_op.allocated_dims		  = ffn_inp->allocated_dims;
				ffn_norm_rms_op.data_type_val = ffn_inp->data_type_val;
				ffn_norm_rms_op.input_ops	  = { ffn_inp };
				ffn_norm_rms_op.depth		  = ffn_inp->depth + 1;
				ffn_inp->dependent_ops.emplace_back(ffn_norm_rms_op.op_id);
				copy_norm_params(ffn_norm_rms_op);
				model.op_cores.emplace_back(std::move(ffn_norm_rms_op));
				core_base_creation_data* ffn_norm_rms = &model.op_cores.back();

				core_base_creation_data ffn_norm_op{};
				ffn_norm_op.type		  = op_type::mul;
				ffn_norm_op.op_id		  = op_id_counter++;
				ffn_norm_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_norm)][block_idXd];
				ffn_norm_op.allocated_dims		  = ffn_norm_rms->allocated_dims;
				ffn_norm_op.data_type_val = ffn_norm_rms->data_type_val;
				ffn_norm_op.input_ops	  = { ffn_norm_rms,  weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_norm.weight") };
				ffn_norm_op.depth		  = std::max(ffn_norm_rms->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_norm.weight")->depth) + 1;
				ffn_norm_rms->dependent_ops.emplace_back(ffn_norm_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_norm.weight")->dependent_ops.emplace_back(ffn_norm_op.op_id);
				model.op_cores.emplace_back(std::move(ffn_norm_op));
				core_base_creation_data* ffn_norm_out = &model.op_cores.back();

				core_base_creation_data ffn_gate_op{};
				ffn_gate_op.type		  = op_type::mul_mat;
				ffn_gate_op.op_id		  = op_id_counter++;
				ffn_gate_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_gate)][block_idXd];
				ffn_gate_op.allocated_dims		  = { feed_forward_length, 2, 1, 1 };
				ffn_gate_op.data_type_val = ffn_norm_out->data_type_val;
				ffn_gate_op.input_ops	  = { ffn_norm_out,  weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_gate.weight") };
				ffn_gate_op.depth		  = std::max(ffn_norm_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_gate.weight")->depth) + 1;
				ffn_norm_out->dependent_ops.emplace_back(ffn_gate_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_gate.weight")->dependent_ops.emplace_back(ffn_gate_op.op_id);
				copy_ffn_params(ffn_gate_op);
				model.op_cores.emplace_back(std::move(ffn_gate_op));
				core_base_creation_data* ffn_gate_out = &model.op_cores.back();

				core_base_creation_data ffn_up_op{};
				ffn_up_op.type			= op_type::mul_mat;
				ffn_up_op.op_id			= op_id_counter++;
				ffn_up_op.name			= model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_up_projection)][block_idXd];
				ffn_up_op.allocated_dims			= { feed_forward_length, 2, 1, 1 };
				ffn_up_op.data_type_val = ffn_norm_out->data_type_val;
				ffn_up_op.input_ops		= { ffn_norm_out,  weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_up.weight") };
				ffn_up_op.depth			= std::max(ffn_norm_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_up.weight")->depth) + 1;
				ffn_norm_out->dependent_ops.emplace_back(ffn_up_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_up.weight")->dependent_ops.emplace_back(ffn_up_op.op_id);
				copy_ffn_params(ffn_up_op);
				model.op_cores.emplace_back(std::move(ffn_up_op));
				core_base_creation_data* ffn_up_out = &model.op_cores.back();

				core_base_creation_data ffn_silu_op{};
				ffn_silu_op.type		  = op_type::silu;
				ffn_silu_op.op_id		  = op_id_counter++;
				ffn_silu_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_activation)][block_idXd];
				ffn_silu_op.allocated_dims		  = ffn_gate_out->allocated_dims;
				ffn_silu_op.data_type_val = ffn_gate_out->data_type_val;
				ffn_silu_op.input_ops	  = { ffn_gate_out };
				ffn_silu_op.depth		  = ffn_gate_out->depth + 1;
				ffn_gate_out->dependent_ops.emplace_back(ffn_silu_op.op_id);
				model.op_cores.emplace_back(std::move(ffn_silu_op));
				core_base_creation_data* ffn_silu_out = &model.op_cores.back();

				core_base_creation_data ffn_gate_par_op{};
				ffn_gate_par_op.type		  = op_type::mul;
				ffn_gate_par_op.op_id		  = op_id_counter++;
				ffn_gate_par_op.name		  = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_gated_output)][block_idXd];
				ffn_gate_par_op.allocated_dims		  = ffn_silu_out->allocated_dims;
				ffn_gate_par_op.data_type_val = ffn_silu_out->data_type_val;
				ffn_gate_par_op.input_ops	  = { ffn_silu_out,  ffn_up_out };
				ffn_gate_par_op.depth		  = std::max(ffn_silu_out->depth, ffn_up_out->depth) + 1;
				ffn_silu_out->dependent_ops.emplace_back(ffn_gate_par_op.op_id);
				ffn_up_out->dependent_ops.emplace_back(ffn_gate_par_op.op_id);
				model.op_cores.emplace_back(std::move(ffn_gate_par_op));
				core_base_creation_data* ffn_gate_par_out = &model.op_cores.back();

				core_base_creation_data ffn_out_op{};
				ffn_out_op.type			 = op_type::mul_mat;
				ffn_out_op.op_id		 = op_id_counter++;
				ffn_out_op.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::ffn_projection)][block_idXd];
				ffn_out_op.allocated_dims			 = { embedding_length, 2, 1, 1 };
				ffn_out_op.data_type_val = ffn_gate_par_out->data_type_val;
				ffn_out_op.input_ops	 = { ffn_gate_par_out,  weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_down.weight") };
				ffn_out_op.depth		 = std::max(ffn_gate_par_out->depth, weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_down.weight")->depth) + 1;
				ffn_gate_par_out->dependent_ops.emplace_back(ffn_out_op.op_id);
				weight_map.at("blk." + std::to_string(block_idXd) + ".ffn_down.weight")->dependent_ops.emplace_back(ffn_out_op.op_id);
				copy_ffn_params(ffn_out_op);
				model.op_cores.emplace_back(std::move(ffn_out_op));
				core_base_creation_data* ffn_out = &model.op_cores.back();

				core_base_creation_data l_out_op{};
				l_out_op.type		   = op_type::add;
				l_out_op.op_id		   = op_id_counter++;
				l_out_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::layer_output)][block_idXd];
				l_out_op.allocated_dims		   = ffn_inp->allocated_dims;
				l_out_op.data_type_val = ffn_inp->data_type_val;
				l_out_op.input_ops	   = { ffn_inp,  ffn_out };
				l_out_op.depth		   = std::max(ffn_inp->depth, ffn_out->depth) + 1;
				ffn_inp->dependent_ops.emplace_back(l_out_op.op_id);
				ffn_out->dependent_ops.emplace_back(l_out_op.op_id);
				model.op_cores.emplace_back(std::move(l_out_op));
				*/
				current_input = &model.op_cores.back();
			}
			/*
			size_t max_depth = std::max_element(model.op_cores.begin(), model.op_cores.end(), [](const core_base_creation_data& lhs, const core_base_creation_data& rhs) {
				return lhs.depth < rhs.depth;
			})->depth;

			core_base_creation_data final_norm_rms_op{};
			final_norm_rms_op.type			= op_type::rms_norm;
			final_norm_rms_op.op_id			= op_id_counter++;
			final_norm_rms_op.name			= model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::final_norm)][0];
			final_norm_rms_op.allocated_dims			= current_input->allocated_dims;
			final_norm_rms_op.data_type_val = current_input->data_type_val;
			final_norm_rms_op.input_ops		= { current_input };
			final_norm_rms_op.depth			= max_depth + 1;
			current_input->dependent_ops.emplace_back(final_norm_rms_op.op_id);
			copy_norm_params(final_norm_rms_op);
			model.op_cores.emplace_back(std::move(final_norm_rms_op));
			core_base_creation_data* final_norm_rms = &model.op_cores.back();

			core_base_creation_data result_norm_op{};
			result_norm_op.type			 = op_type::mul;
			result_norm_op.op_id		 = op_id_counter++;
			result_norm_op.name			 = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::output_norm)][0];
			result_norm_op.allocated_dims			 = final_norm_rms->allocated_dims;
			result_norm_op.data_type_val = final_norm_rms->data_type_val;
			result_norm_op.input_ops	 = { final_norm_rms,  weight_map.at("output_norm.weight") };
			result_norm_op.depth		 = max_depth + 1;
			final_norm_rms->dependent_ops.emplace_back(result_norm_op.op_id);
			weight_map.at("output_norm.weight")->dependent_ops.emplace_back(result_norm_op.op_id);
			model.op_cores.emplace_back(std::move(result_norm_op));
			core_base_creation_data* result_norm = &model.op_cores.back();

			core_base_creation_data result_output_op{};
			result_output_op.type		   = op_type::mul_mat;
			result_output_op.op_id		   = op_id_counter++;
			result_output_op.name		   = model_arch_traits<model_arch::llama>::op_names[static_cast<size_t>(llama_op_names::logits_output)][0];
			result_output_op.allocated_dims		   = { vocab_size, 1, 1, 1 };
			result_output_op.data_type_val = result_norm->data_type_val;
			result_output_op.input_ops	   = { result_norm,  weight_map.at("output.weight") };
			result_output_op.depth		   = max_depth + 1;
			result_norm->dependent_ops.emplace_back(result_output_op.op_id);
			weight_map.at("output.weight")->dependent_ops.emplace_back(result_output_op.op_id);
			model.op_cores.emplace_back(std::move(result_output_op));
			*/
			model.op_cores.shrink_to_fit();
		}

		RT_TM_FORCE_INLINE static model_graph<config> parse_model(std::string_view path) {
			std::string data_val{ file_loader<config.exceptions>{ path } };
			model_graph<global_config{}> return_value{};
			gguf_file_t gguf_file{};
			string_iterator ptr{};
			ptr.first_index	 = data_val.data();
			ptr.length		 = data_val.size();
			gguf_file.header = value_reader<gguf_header_t>::gather_value(ptr);
			for (size_t x = 0; x < gguf_file.header.tensor_count; ++x) {
				gguf_file.tensor_infos.emplace_back(value_reader<gguf_tensor_info_t>::gather_value(ptr));
			}
			auto calculate_tensor_size = [](const gguf_tensor_info_t& tensor) -> size_t {
				core_base_creation_data temp_core{};
				temp_core.data_type_val = tensor.type;
				for (size_t i = 0; i < tensor.n_dimensions; ++i) {
					temp_core.allocated_dims[i] = tensor.dimensions[i];
				}
				return temp_core.core_total_byte_size();
			};

			size_t total_tensor_bytes = 0;
			size_t max_tensor_end	  = 0;
			for (const auto& tensor: gguf_file.tensor_infos) {
				size_t tensor_size = calculate_tensor_size(tensor);
				total_tensor_bytes += tensor_size;
				size_t tensor_end = tensor.offset + tensor_size;
				max_tensor_end	  = std::max(max_tensor_end, tensor_end);
			}
			return_value.leaf_core_data.init(total_tensor_bytes);

			size_t tensor_data_start = data_val.size() - max_tensor_end;
			uint64_t alignment{ 32 };
			gather_scalar("alignment", alignment, gguf_file.header.metadata_kv);
			return_value.hparams		  = value_reader<hyper_parameters<model_arch::llama>, model_arch::llama>::gather_value(gguf_file.header.metadata_kv);
			return_value.tokenizer_params = value_reader<tokenizer_parameters<model_arch::llama>, model_arch::llama>::gather_value(gguf_file.header.metadata_kv);
			sort_tensor_infos(gguf_file.tensor_infos);
			for (size_t x = 0; x < gguf_file.header.tensor_count; ++x) {
				core_base_creation_data new_core{};
				new_core.name = model_arch_traits<model_arch::llama>::op_names[string_to_tensor_name<model_arch::llama>::impl(gguf_file.tensor_infos[x].name)]
																			  [extract_layer_number(gguf_file.tensor_infos[x].name)];
				new_core.type		   = op_type::noop;
				new_core.depth		   = 0;
				new_core.op_id		   = x;
				new_core.data_type_val = gguf_file.tensor_infos[x].type;
				for (size_t y = 0; y < gguf_file.tensor_infos[x].n_dimensions; ++y) {
					new_core.allocated_dims[y] = gguf_file.tensor_infos[x].dimensions[y];
					new_core.allocated_dims[y] = gguf_file.tensor_infos[x].dimensions[y];
				}
				size_t current_size{ new_core.core_total_byte_size() };
				size_t absolute_offset = tensor_data_start + gguf_file.tensor_infos[x].offset;
				auto* ptr_new		   = return_value.leaf_core_data.claim_memory(current_size);
				new_core.data		   = ptr_new;
				std::memcpy(ptr_new, data_val.data() + absolute_offset, current_size);
				return_value.op_cores.emplace_back(new_core);
			}
			generate_ops(return_value);
			std::cout << "CURRENT COUNT: " << return_value.op_cores.size() << std::endl;
			for (size_t x = 0; x < return_value.op_cores.size(); ++x) {
				debugging_io<false, core_base_creation_data>::load_and_compare_tensors(return_value.op_cores[x]);
			}
			return return_value;
		}
	};
}
