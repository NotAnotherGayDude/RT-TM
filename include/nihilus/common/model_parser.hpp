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

#include <nihilus/common/type_traits.hpp>
#include <nihilus/common/model_graph_data.hpp>
#include <nihilus/common/debugging_io.hpp>
#include <nihilus/common/model_traits.hpp>
#include <unordered_set>
#include <variant>
#include <fstream>
#include <regex>
#include <map>
#include <bit>

namespace nihilus {

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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <filesystem>
#include <stdexcept>

#ifdef NIHILUS_PLATFORM_WINDOWS
	#include <windows.h>
#else
	#include <sys/mman.h>
	#include <fcntl.h>
	#include <unistd.h>
#endif

	class mapped_file {
	  public:
		void* aligned_ptr  = nullptr;
		void* base_ptr	   = nullptr;
		size_t mapped_size = 0;
		size_t file_size   = 0;

		static constexpr size_t alignment = 64;

		mapped_file(std::string_view path) {
			namespace fs = std::filesystem;
			file_size	 = fs::file_size(path);
			mapped_size	 = file_size + alignment;

#if NIHILUS_PLATFORM_WINDOWS
			HANDLE file = CreateFileA(path.data(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
			if (file == INVALID_HANDLE_VALUE)
				throw std::runtime_error("Failed to open file");

			HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
			if (!mapping)
				throw std::runtime_error("Failed to create file mapping");

			void* raw = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, mapped_size);
			if (!raw)
				throw std::runtime_error("Failed to map view of file");

			base_ptr	= raw;
			aligned_ptr = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(alignment - 1));

			CloseHandle(file);
			CloseHandle(mapping);
#else
			int fd = open(path.data(), O_RDONLY);
			if (fd < 0)
				throw std::runtime_error("Failed to open file");

			void* raw = mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
			if (raw == MAP_FAILED)
				throw std::runtime_error("mmap failed");

			base_ptr	= raw;
			aligned_ptr = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(alignment - 1));

			close(fd);
#endif
		}

		~mapped_file() {
			if (!base_ptr)
				return;

#ifdef _WIN32
			UnmapViewOfFile(base_ptr);
#else
			munmap(base_ptr, mapped_size);
#endif
		}

		template<typename T = uint8_t> const T* data() const {
			return reinterpret_cast<const T*>(aligned_ptr);
		}

		size_t size() const {
			return file_size;
		}
	};


	struct stream_iterator {
		std::ifstream* stream  = nullptr;
		std::streambuf* buffer = nullptr;
		uint64_t current_index = 0;
		uint64_t length		   = 0;
		bool valid			   = false;

		NIHILUS_FORCE_INLINE stream_iterator(std::ifstream& s) : stream(&s), buffer(s.rdbuf()), current_index(0), length(0), valid(false) {
			if (stream && stream->is_open()) {
				auto pos = stream->tellg();
				stream->seekg(0, std::ios::end);
				length = static_cast<uint64_t>(stream->tellg());
				stream->seekg(pos);
				valid = stream->good();
			}
		}

		template<typename value_type> NIHILUS_FORCE_INLINE stream_iterator& advance(uint64_t n = 1) {
			static_assert(std::is_trivially_copyable_v<value_type>);
			uint64_t offset = sizeof(value_type) * n;
			if (valid && has_bytes<uint8_t>(offset)) {
				stream->seekg(offset, std::ios::cur);
				current_index += offset;
			} else {
				valid = false;
			}
			return *this;
		}

		template<typename value_type> NIHILUS_FORCE_INLINE value_type read() {
			static_assert(std::is_trivially_copyable_v<value_type>);
			value_type v{};
			if (valid && has_bytes<value_type>()) {
				auto read_bytes = buffer->sgetn(reinterpret_cast<char*>(&v), sizeof(value_type));
				current_index += static_cast<uint64_t>(read_bytes);
				valid = (read_bytes == sizeof(value_type));
			} else {
				valid = false;
			}
			return v;
		}

		NIHILUS_FORCE_INLINE void read_bytes_to_pointer(void* dest_ptr, uint64_t byte_count) {
			if (valid && has_bytes<uint8_t>(byte_count)) {
				auto read_bytes = buffer->sgetn(reinterpret_cast<char*>(dest_ptr), static_cast<int64_t>(byte_count));
				current_index += static_cast<uint64_t>(read_bytes);
				valid = (read_bytes == static_cast<int64_t>(byte_count));
			} else {
				valid = false;
			}
		}

		template<typename value_type = uint8_t> NIHILUS_FORCE_INLINE bool has_bytes(size_t size = sizeof(value_type)) const {
			return (current_index + size <= length);
		}
	};

	template<typename value_type, auto...> struct value_reader {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			if (input.has_bytes<value_type>()) {
				return input.read<value_type>();
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
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
		NIHILUS_FORCE_INLINE static std::string gather_value(stream_iterator& input) {
			uint64_t length = value_reader<uint64_t>::gather_value(input);
			if (!input.has_bytes<char>(length)) {
				throw std::runtime_error("Sorry, but that index is out of range!");
			}
			std::string result(length, '\0');
			result.resize(length);
			input.read_bytes_to_pointer(result.data(), length);
			return result;
		}
	};

	template<> struct value_reader<gguf_array_t> {
		NIHILUS_FORCE_INLINE static gguf_array_t gather_value(stream_iterator& input);
	};

	template<> struct value_reader<gguf_metadata_value_variant> {
		NIHILUS_INLINE static gguf_metadata_value_variant gather_value(stream_iterator& input, gguf_metadata_value_type type) {
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

	gguf_array_t value_reader<gguf_array_t>::gather_value(stream_iterator& input) {
		gguf_metadata_value_type type{ value_reader<gguf_metadata_value_type>::gather_value(input) };
		uint64_t length{ value_reader<uint64_t>::gather_value(input) };
		constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
		if (length > MAX_ARRAY_LENGTH) {
			throw std::runtime_error{ "Array length exceeds maximum allowed size!" };
		}
		gguf_array_t value{};
		value.type = type;
		value.array.reserve(length);
		for (uint64_t x = 0; x < length; ++x) {
			value.array.emplace_back(value_reader<gguf_metadata_value_variant>::gather_value(input, type));
		}
		return value;
	}

	struct gguf_metadata_kv_t;

	struct gguf_metadata_kv_t {
		gguf_metadata_value_type value_type{};

		gguf_metadata_value_t value{};

		NIHILUS_FORCE_INLINE operator bool() const {
			return std::get<bool>(value.value);
		}

		NIHILUS_FORCE_INLINE operator int64_t() const {
			return std::get<int64_t>(value.value);
		}

		NIHILUS_FORCE_INLINE operator uint64_t() const {
			return std::get<uint64_t>(value.value);
		}

		NIHILUS_FORCE_INLINE operator gguf_string_t() const {
			return std::get<gguf_string_t>(value.value);
		}

		NIHILUS_FORCE_INLINE operator gguf_array_t() const {
			return *std::get<gguf_array_t*>(value.value);
		}

		NIHILUS_FORCE_INLINE operator float() const {
			return std::get<float>(value.value);
		}

		NIHILUS_FORCE_INLINE operator double() const {
			return std::get<double>(value.value);
		}
	};

	template<> struct value_reader<gguf_metadata_kv_t> {
		NIHILUS_FORCE_INLINE static gguf_metadata_kv_t gather_value(stream_iterator& input) {
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

	template<typename value_type> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, value_type& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<value_type>(v)) {
			out = std::get<value_type>(v);
		}
	};

	template<typename value_type>
	NIHILUS_FORCE_INLINE void gather_array(const std::string& key, std::vector<value_type>& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
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

	NIHILUS_FORCE_INLINE void print_variant(auto variant) {
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

	template<> struct value_reader<construction_parameters<model_arch::llama>, model_arch::llama> {
		NIHILUS_FORCE_INLINE static construction_parameters<model_arch::llama> gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
			construction_parameters<model_arch::llama> value{};
			std::string architecture{};
			if (metadata_kv.contains("general.architecture")) {
				architecture = metadata_kv.at("general.architecture").operator gguf_string_t();
			}
			gather_scalar(architecture + ".rope.dimension_count", value.rope_dimension_count, metadata_kv);
			gather_scalar(architecture + ".feed_forward_length", value.feed_forward_length, metadata_kv);
			gather_scalar(architecture + ".embedding_length", value.embedding_length, metadata_kv);
			gather_scalar(architecture + ".context_length", value.context_length, metadata_kv);
			gather_scalar(architecture + ".attention.head_count_kv", value.head_count_kv, metadata_kv);
			gather_scalar(architecture + ".block_count", value.block_count, metadata_kv);
			gather_scalar(architecture + ".attention.head_count", value.head_count, metadata_kv);
			gather_scalar(architecture + ".vocab_size", value.vocab_size, metadata_kv);
			gather_scalar(architecture + ".rope.type", value.rope_type, metadata_kv);
			gather_scalar(architecture + ".expert_count", value.n_expert, metadata_kv);
			gather_scalar(architecture + ".expert_used_count", value.n_expert_used, metadata_kv);
			gather_scalar(architecture + ".rope.freq_base", value.rope_freq_base, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.factor", value.rope_freq_scale, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.attn_factor", value.rope_attn_factor, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.beta_fast", value.rope_beta_fast, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.beta_slow", value.rope_beta_slow, metadata_kv);
			gather_scalar(architecture + ".attention.layer_norm_rms_epsilon", value.rms_norm_epsilon, metadata_kv);
			gather_scalar(architecture + ".attention.scale", value.f_attention_scale, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.ext_factor", value.rope_ext_factor, metadata_kv);

			return value;
		}
	};

	template<> struct value_reader<tokenizer_parameters<model_arch::llama>, model_arch::llama> {
		NIHILUS_FORCE_INLINE static tokenizer_parameters<model_arch::llama> gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
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
		NIHILUS_FORCE_INLINE static gguf_header_t gather_value(stream_iterator& input) {
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
			for (uint64_t x = 0; x < value.metadata_kv_count; ++x) {
				std::string new_string		  = value_reader<gguf_string_t>::gather_value(input);
				value.metadata_kv[new_string] = value_reader<gguf_metadata_kv_t>::gather_value(input);
			}
			return value;
		}
	};

	template<model_arch arch> struct string_to_tensor_name;

	template<> struct string_to_tensor_name<model_arch::llama> {
		NIHILUS_FORCE_INLINE static llama_op_types impl(std::string_view input) noexcept {
			if (input == "token_embd.weight")
				return llama_op_types::token_embd_weight;
			if (input == "rope_freqs.weight")
				return llama_op_types::rope_freqs_weight;
			if (input == "output_norm.weight")
				return llama_op_types::output_norm_weight;
			if (input == "output.weight")
				return llama_op_types::output_weight;

			if (input.find(".attn_q.weight") != std::string_view::npos)
				return llama_op_types::attn_q_weight;
			if (input.find(".attn_norm.weight") != std::string_view::npos)
				return llama_op_types::attn_norm_weight;

			if (input.starts_with("blk.") && input.ends_with(".weight")) {
				auto second_dot = input.find('.', 4);
				if (second_dot != std::string_view::npos) {
					auto suffix = input.substr(second_dot + 1);

					if (suffix == "attn_q.weight")
						return llama_op_types::attn_q_weight;
					if (suffix == "attn_norm.weight")
						return llama_op_types::attn_norm;
					if (suffix == "attn_k.weight")
						return llama_op_types::attn_k_weight;
					if (suffix == "attn_v.weight")
						return llama_op_types::attn_v_weight;
					if (suffix == "ffn_down.weight")
						return llama_op_types::ffn_down_weight;
					if (suffix == "ffn_gate.weight")
						return llama_op_types::ffn_gate_weight;
					if (suffix == "attn_output.weight")
						return llama_op_types::attn_output_weight;
					if (suffix == "ffn_norm.weight")
						return llama_op_types::ffn_norm_weight;
					if (suffix == "ffn_up.weight")
						return llama_op_types::ffn_up_weight;
				}
			}

			return llama_op_types::count;
		}
	};

	template<> struct value_reader<core_base_creation_data> {
		NIHILUS_FORCE_INLINE static core_base_creation_data gather_value(stream_iterator& input) {
			core_base_creation_data value{};
			value.name						  = value_reader<gguf_string_t>::gather_value(input);
			value.n_dimensions				  = value_reader<uint32_t>::gather_value(input);
			constexpr uint32_t MAX_DIMENSIONS = 8;
			if (value.n_dimensions > MAX_DIMENSIONS) {
				throw std::runtime_error{ "Tensor dimensions exceed maximum!" };
			}
			for (uint64_t x = 0; x < value.n_dimensions; ++x) {
				uint64_t dim					= value_reader<uint64_t>::gather_value(input);
				constexpr uint64_t MAX_DIM_SIZE = 1ULL << 32;
				if (dim > MAX_DIM_SIZE) {
					throw std::runtime_error{ "Tensor dimension size too large!" };
				}
				value.dimensions[x] = dim;
			}
			value.type	 = static_cast<data_type>(value_reader<uint32_t>::gather_value(input));
			value.offset = value_reader<uint64_t>::gather_value(input);
			return value;
		}
	};

	NIHILUS_FORCE_INLINE bool operator<(const core_base_creation_data& lhs, const core_base_creation_data& rhs) noexcept {
		const std::string& lhs_name{ lhs.name };
		const std::string& rhs_name{ rhs.name };
		if (lhs_name.find_first_of("1234567890") != std::string::npos && rhs_name.find_first_of("1234567890") != std::string::npos) {
			uint64_t lhs_offset{ lhs_name.find_first_of("1234567890") };
			uint64_t rhs_offset{ rhs_name.find_first_of("1234567890") };
			std::string lhs_val_raw{ lhs_name.substr(lhs_offset, lhs_name.find_first_not_of("1234567890", lhs_offset) - lhs_offset + 1) };
			std::string rhs_val_raw{ rhs_name.substr(rhs_offset, rhs_name.find_first_not_of("1234567890", rhs_offset) - rhs_offset + 1) };
			auto* lhs_ptr_01 = lhs_val_raw.data();
			auto* lhs_ptr_02 = lhs_val_raw.data() + lhs_val_raw.size();
			uint64_t lhs_val{ std::strtoull(lhs_ptr_01, &lhs_ptr_02, 10) };
			auto* rhs_ptr_01 = rhs_val_raw.data();
			auto* rhs_ptr_02 = rhs_val_raw.data() + rhs_val_raw.size();
			uint64_t rhs_val{ std::strtoull(rhs_ptr_01, &rhs_ptr_02, 10) };
			return lhs_val < rhs_val;
		} else {
			return lhs_name < rhs_name;
		}
	}

	NIHILUS_FORCE_INLINE void sort_tensor_infos(std::vector<core_base_creation_data>& tensor_infos) noexcept {
		std::sort(tensor_infos.begin(), tensor_infos.end(), std::less<core_base_creation_data>{});
	}

	NIHILUS_FORCE_INLINE constexpr uint64_t parse_number(std::string_view str) noexcept {
		uint64_t result = 0;
		for (char c: str) {
			if (c >= '0' && c <= '9') {
				result = result * 10 + (c - '0');
			} else {
				break;
			}
		}
		return result;
	}

	NIHILUS_FORCE_INLINE uint64_t extract_layer_number(std::string_view name) noexcept {
		if NIHILUS_LIKELY (name[0] == 'c' && name.starts_with("cache_")) {
			for (uint64_t i = 7; i < name.size(); ++i) {
				if (name[i] == 'l' && i + 1 < name.size()) {
					return parse_number(name.substr(i + 1));
				}
			}
			return 0;
		}
		if NIHILUS_LIKELY (name[0] == 'b' && name.starts_with("blk.")) {
			uint64_t start = 4;
			uint64_t end   = name.find('.', start);
			if (end != std::string_view::npos) {
				return parse_number(name.substr(start, end - start));
			}
		}

		return 0;
	}

	struct gguf_file_t {
		std::vector<core_base_creation_data> tensor_infos{};
		std::vector<uint8_t> tensor_data{};
		std::vector<uint8_t> _padding{};
		gguf_header_t header{};
	};

	NIHILUS_FORCE_INLINE uint64_t align_offset(uint64_t offset, uint64_t alignment = 1) {
		alignment = alignment == 0 ? 1 : alignment;
		return offset + (alignment - (offset % alignment)) % alignment;
	}

	template<model_config config, model_arch arch, model_format type> struct model_parser_impl;

	template<model_config config> struct model_parser_impl<config, model_arch::llama, model_format::gguf> {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		using op_type_type		= typename decltype(config)::op_type_type;
		static_assert((std::endian::native == std::endian::little), "Sorry, but big-endian is not yet supported by the library");

		NIHILUS_FORCE_INLINE static model_graph_data<config> parse_model(std::string_view path, array<array<void*, model_traits_type::block_count>, op_type_type::count>& data) {
			std::ifstream file(static_cast<std::string>(path), std::ios::binary);
			stream_iterator ptr{ file };
			model_graph_data<config> return_value{};
			gguf_file_t gguf_file{};
			gguf_file.header = value_reader<gguf_header_t>::gather_value(ptr);
			for (uint64_t x = 0; x < gguf_file.header.tensor_count; ++x) {
				gguf_file.tensor_infos.emplace_back(value_reader<core_base_creation_data>::gather_value(ptr));
			}

			uint64_t total_tensor_bytes = 0;
			uint64_t max_tensor_end		= 0;
			for (const auto& tensor: gguf_file.tensor_infos) {
				uint64_t tensor_size = tensor.core_total_byte_size();
				total_tensor_bytes += tensor_size;
				uint64_t tensor_end = tensor.offset + tensor_size;
				max_tensor_end		= std::max(max_tensor_end, tensor_end);
			}

			uint64_t tensor_data_start = 0;
			uint64_t alignment{ 32 };
			gather_scalar("alignment", alignment, gguf_file.header.metadata_kv);
			return_value.cparams		  = value_reader<construction_parameters<model_arch::llama>, model_arch::llama>::gather_value(gguf_file.header.metadata_kv);
			return_value.tokenizer_params = value_reader<tokenizer_parameters<model_arch::llama>, model_arch::llama>::gather_value(gguf_file.header.metadata_kv);
			sort_tensor_infos(gguf_file.tensor_infos);
			for (uint64_t x = 0; x < gguf_file.tensor_infos.size(); ++x) {
				//std::cout << "TOTAL REQUIRED BYTES (PRE): " << gguf_file.tensor_infos[x].core_total_byte_size()
				//<< ", FOR TYPE: " << static_cast<size_t>(string_to_tensor_name<model_arch::llama>::impl(gguf_file.tensor_infos[x].name));
				//std::cout << ", NAME: " << gguf_file.tensor_infos[x].name;
				//std::cout << ", DIMS: " << gguf_file.tensor_infos[x].dimensions << std::endl;
				//ptr.read_bytes_to_pointer(
				//data[string_to_tensor_name<model_arch::llama>::impl(gguf_file.tensor_infos[x].name)][extract_layer_number(gguf_file.tensor_infos[x].name)],
				//gguf_file.tensor_infos[x].core_total_byte_size());
			}
			return return_value;
		}
	};

	template<model_config config> struct model_parser;

	template<model_config config> struct model_parser {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		using op_type_type		= typename decltype(config)::op_type_type;

		NIHILUS_FORCE_INLINE static model_graph_data<config> parse_model(std::string_view path, array<array<void*, model_traits_type::block_count>, op_type_type::count>& data) {
			return model_parser_impl<config, config.arch, config.format>::parse_model(path, data);
		}
	};
}
