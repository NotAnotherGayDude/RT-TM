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

#include <rt_tm/common/arch_traits.hpp>
#include <rt_tm/common/core_base.hpp>
#include <rt_tm/common/common.hpp>
#include <filesystem>
#include <stdexcept>
#include <charconv>
#include <cstdint>
#include <fstream>

namespace rt_tm {

	template<bool exceptions> class file_loader {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				if constexpr (exceptions) {
					throw std::runtime_error("File does not exist: " + filePath.string());
				} else {
					std::cerr << "File does not exist: " + filePath.string() << std::endl;
				}
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				if constexpr (exceptions) {
					throw std::runtime_error("Failed to open file: " + filePath.string());
				} else {
					std::cerr << "Failed to open file: " + filePath.string() << std::endl;
				}
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					if constexpr (exceptions) {
						throw std::runtime_error("Failed to read file: " + filePath.string());
					} else {
						std::cerr << "Failed to read file: " + filePath.string() << std::endl;
					}
				}
			}
		}

		operator const std::string&() const noexcept {
			return contents;
		}

		uint64_t size() const noexcept {
			return contents.size();
		}

	  private:
		std::string contents;
	};

	template<bool exceptions> class file_saver {
	  public:
		file_saver(const std::filesystem::path& path, const void* data, uint64_t size) {
			if (!data || size == 0) {
				if constexpr (exceptions) {
					throw std::runtime_error("Cannot save null or empty data to file: " + path.string());
				} else {
					std::cerr << "Cannot save null or empty data to file: " + path.string() << std::endl;
				}
			}

			std::ofstream file(path, std::ios::binary | std::ios::trunc);
			if (!file) {
				if constexpr (exceptions) {
					throw std::runtime_error("Failed to open file for writing: " + path.string());
				} else {
					std::cerr << "Failed to open file for writing: " + path.string() << std::endl;
				}
			}

			file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
			if (!file) {
				if constexpr (exceptions) {
					throw std::runtime_error("Failed to write data to file: " + path.string());
				} else {
					std::cerr << "Failed to write data to file: " + path.string() << std::endl;
				}
			}
		}
	};

	std::string map_rt_tm_to_ggml(llama_op_types, uint64_t) {
		return {};
		//arch_traits<model_arch::llama>::tensor_names[static_cast<uint64_t>(rt_tm_enum)][layer_index].operator const char*();
	}

	std::string convert_rt_tm_name_to_ggml(std::string_view rt_tm_name) {
		uint64_t dash_pos	   = rt_tm_name.find_last_of('-');
		uint64_t layer_index = 0;

		if (dash_pos != std::string_view::npos) {
			std::string_view number_part = rt_tm_name.substr(dash_pos + 1);
			auto result					 = std::from_chars(number_part.data(), number_part.data() + number_part.size(), layer_index);
			if (result.ec != std::errc{}) {
				layer_index = 0;
			}
		}

		std::string_view base_name = (dash_pos != std::string_view::npos) ? rt_tm_name.substr(0, dash_pos) : rt_tm_name;
		if (base_name == "inp_tokens")
			return map_rt_tm_to_ggml(llama_op_types::inp_tokens, layer_index);
		return static_cast<std::string>(rt_tm_name);
	}

	struct intermediary_tensor {
		static constexpr uint64_t tensor_len_size{ sizeof(uint64_t) };
		static constexpr uint64_t name_len_size{ sizeof(uint64_t) };
		static constexpr uint64_t dims_size{ sizeof(uint64_t) * 4 };
		static constexpr uint64_t type_size{ sizeof(uint32_t) };
		static constexpr uint64_t op_size{ sizeof(uint32_t) };
		std::vector<std::string> input_names{};
		array<uint64_t, 4> dims{};
		std::string name{};
		data_type type{};
		kernel_type op{};

		intermediary_tensor() noexcept = default;
		/*
		intermediary_tensor(const core_base& other) {
			//uint64_t nbytes{ other.core_total_byte_size() };
			for (uint64_t x = 0; x < 4; ++x) {
				dims[x] = other.allocated_dims[x];
			}
			op	 = other.type;
			type = other.data_type_val;
			name = { other.name };
		}
		*/
		intermediary_tensor(const core_base_creation_data& other) {
			//uint64_t nbytes{ other.core_total_byte_size() };
			for (uint64_t x = 0; x < 4; ++x) {
				dims[x] = other.allocated_dims[x];
			}
			op	 = other.type;
			type = other.data_type_val;
			name = { other.name };
		}

		RT_TM_FORCE_INLINE friend bool operator==(const intermediary_tensor& lhs, const intermediary_tensor& rhs) {
			std::stringstream stream{};
			bool same{ true };
			std::cout << "Correct Name: " << lhs.name << std::endl;
			std::cout << "Current Name: " << rhs.name << std::endl;
			if (lhs.op != rhs.op) {
				stream << "Different op-type!" << std::endl;
				stream << "Correct op-type: " << static_cast<int32_t>(lhs.op) << std::endl;
				stream << "Current op-type: " << static_cast<int32_t>(rhs.op) << std::endl;
				same = false;
			}
			if (lhs.type != rhs.type) {
				stream << "Different type!" << std::endl;
				stream << "Correct type: " << static_cast<int32_t>(lhs.type) << std::endl;
				stream << "Current type: " << static_cast<int32_t>(rhs.type) << std::endl;
				same = false;
			}
			if (lhs.dims != rhs.dims) {
				stream << "Different dims!" << std::endl;
				stream << "Correct Dims: " << lhs.dims << std::endl;
				stream << "Current Dims: " << rhs.dims << std::endl;
				same = false;
			}
			if (!same) {
				std::cerr << "For Tensor: " << rhs.name << std::endl;
				std::cerr << stream.str() << std::endl;
			}
			return same;
		}

		/*
		intermediary_tensor(const ggml_tensor& other) {
			uint64_t nbytes{ ggml_nbytes(&other) };
			data.resize(nbytes);
			std::memcpy(data.data(), other.data, nbytes);
			for (uint64_t x = 0; x < 4; ++x) {
				dims[x] = other.ne[x];
			}
			type = static_cast<data_type>(other.type);
			name = std::string{other.name};
		}*/
	};

	enum ggml_op {
		GGML_OP_NONE = 0,

		GGML_OP_DUP,
		GGML_OP_ADD,
		GGML_OP_ADD1,
		GGML_OP_ACC,
		GGML_OP_SUB,
		GGML_OP_MUL,
		GGML_OP_DIV,
		GGML_OP_SQR,
		GGML_OP_SQRT,
		GGML_OP_LOG,
		GGML_OP_SIN,
		GGML_OP_COS,
		GGML_OP_SUM,
		GGML_OP_SUM_ROWS,
		GGML_OP_MEAN,
		GGML_OP_ARGMAX,
		GGML_OP_COUNT_EQUAL,
		GGML_OP_REPEAT,
		GGML_OP_REPEAT_BACK,
		GGML_OP_CONCAT,
		GGML_OP_SILU_BACK,
		GGML_OP_NORM,
		GGML_OP_RMS_NORM,
		GGML_OP_RMS_NORM_BACK,
		GGML_OP_GROUP_NORM,
		GGML_OP_L2_NORM,

		GGML_OP_MUL_MAT,
		GGML_OP_MUL_MAT_ID,
		GGML_OP_OUT_PROD,

		GGML_OP_SCALE,
		GGML_OP_SET,
		GGML_OP_CPY,
		GGML_OP_CONT,
		GGML_OP_RESHAPE,
		GGML_OP_VIEW,
		GGML_OP_PERMUTE,
		GGML_OP_TRANSPOSE,
		GGML_OP_GET_ROWS,
		GGML_OP_GET_ROWS_BACK,
		GGML_OP_DIAG,
		GGML_OP_DIAG_MASK_INF,
		GGML_OP_DIAG_MASK_ZERO,
		GGML_OP_SOFT_MAX,
		GGML_OP_SOFT_MAX_BACK,
		GGML_OP_ROPE,
		GGML_OP_ROPE_BACK,
		GGML_OP_CLAMP,
		GGML_OP_CONV_TRANSPOSE_1D,
		GGML_OP_IM2COL,
		GGML_OP_IM2COL_BACK,
		GGML_OP_CONV_TRANSPOSE_2D,
		GGML_OP_POOL_1D,
		GGML_OP_POOL_2D,
		GGML_OP_POOL_2D_BACK,
		GGML_OP_UPSCALE,
		GGML_OP_PAD,
		GGML_OP_PAD_REFLECT_1D,
		GGML_OP_ARANGE,
		GGML_OP_TIMESTEP_EMBEDDING,
		GGML_OP_ARGSORT,
		GGML_OP_LEAKY_RELU,

		GGML_OP_FLASH_ATTN_EXT,
		GGML_OP_FLASH_ATTN_BACK,
		GGML_OP_SSM_CONV,
		GGML_OP_SSM_SCAN,
		GGML_OP_WIN_PART,
		GGML_OP_WIN_UNPART,
		GGML_OP_GET_REL_POS,
		GGML_OP_ADD_REL_POS,
		GGML_OP_RWKV_WKV6,
		GGML_OP_GATED_LINEAR_ATTN,
		GGML_OP_RWKV_WKV7,

		GGML_OP_UNARY,

		GGML_OP_MAP_UNARY,
		GGML_OP_MAP_BINARY,

		GGML_OP_MAP_CUSTOM1_F32,
		GGML_OP_MAP_CUSTOM2_F32,
		GGML_OP_MAP_CUSTOM3_F32,

		GGML_OP_MAP_CUSTOM1,
		GGML_OP_MAP_CUSTOM2,
		GGML_OP_MAP_CUSTOM3,

		GGML_OP_CROSS_ENTROPY_LOSS,
		GGML_OP_CROSS_ENTROPY_LOSS_BACK,
		GGML_OP_OPT_STEP_ADAMW,

		GGML_OP_COUNT,
	};

	enum kernel_type convert_ggml_op_to_op_type(enum ggml_op) {
		return {};
	}

	intermediary_tensor parse_tensor_from_string(const std::string& file_contents) {
		intermediary_tensor tensor{};
		uint64_t offset = 0;

		constexpr uint64_t min_header_size =
			intermediary_tensor::type_size + intermediary_tensor::op_size + intermediary_tensor::dims_size + intermediary_tensor::name_len_size + sizeof(uint64_t);

		if (file_contents.size() < min_header_size) {
			throw std::runtime_error("File contents too small for valid tensor");
		}

		if (offset + intermediary_tensor::type_size > file_contents.size()) {
			throw std::runtime_error("Not enough data for type field");
		}
		uint32_t temp_type{};
		std::memcpy(&temp_type, file_contents.data() + offset, intermediary_tensor::type_size);
		offset += intermediary_tensor::type_size;
		tensor.type = static_cast<data_type>(temp_type);

		if (offset + intermediary_tensor::op_size > file_contents.size()) {
			throw std::runtime_error("Not enough data for operation field");
		}
		ggml_op temp_op_type{};
		std::memcpy(&temp_op_type, file_contents.data() + offset, intermediary_tensor::op_size);
		offset += intermediary_tensor::op_size;
		tensor.op = convert_ggml_op_to_op_type(temp_op_type);
		if (offset + intermediary_tensor::dims_size > file_contents.size()) {
			throw std::runtime_error("Not enough data for dimensions");
		}
		std::memcpy(tensor.dims.data(), file_contents.data() + offset, intermediary_tensor::dims_size);
		offset += intermediary_tensor::dims_size;

		uint64_t name_length = 0;
		if (offset + intermediary_tensor::name_len_size > file_contents.size()) {
			throw std::runtime_error("Not enough data for name length");
		}
		std::memcpy(&name_length, file_contents.data() + offset, intermediary_tensor::name_len_size);
		offset += intermediary_tensor::name_len_size;

		if (name_length > 10000) {
			throw std::runtime_error("Suspiciously large name length: " + std::to_string(name_length));
		}

		if (name_length > 0) {
			if (offset + name_length > file_contents.size()) {
				throw std::runtime_error("Not enough data for tensor name");
			}
			tensor.name.resize(name_length);
			std::memcpy(tensor.name.data(), file_contents.data() + offset, name_length);
			offset += name_length;
		}

		uint64_t input_count = 0;
		if (offset + sizeof(uint64_t) > file_contents.size()) {
			throw std::runtime_error("Not enough data for input count");
		}
		std::memcpy(&input_count, file_contents.data() + offset, sizeof(uint64_t));
		offset += sizeof(uint64_t);

		if (input_count > 1000) {
			throw std::runtime_error("Suspiciously large input count: " + std::to_string(input_count));
		}

		tensor.input_names.reserve(input_count);
		for (uint64_t i = 0; i < input_count; ++i) {
			uint64_t input_name_length = 0;
			if (offset + intermediary_tensor::name_len_size > file_contents.size()) {
				throw std::runtime_error("Not enough data for input name length at index " + std::to_string(i));
			}
			std::memcpy(&input_name_length, file_contents.data() + offset, intermediary_tensor::name_len_size);
			offset += intermediary_tensor::name_len_size;

			if (input_name_length > 10000) {
				throw std::runtime_error("Suspiciously large input name length: " + std::to_string(input_name_length));
			}

			std::string input_name{};
			if (input_name_length > 0) {
				if (offset + input_name_length > file_contents.size()) {
					throw std::runtime_error("Not enough data for input name at index " + std::to_string(i));
				}
				input_name.resize(input_name_length);
				std::memcpy(input_name.data(), file_contents.data() + offset, input_name_length);
				offset += input_name_length;
			}

			tensor.input_names.push_back(std::move(input_name));
		}

		return tensor;
	}

	intermediary_tensor parse_tensor_from_string_safe(const std::string& file_contents, bool strict_validation = true) {
		try {
			auto tensor = parse_tensor_from_string(file_contents);

			if (strict_validation) {
				if (tensor.name.empty()) {
					std::cerr << "Warning: Tensor has empty name" << std::endl;
				}

				for (uint64_t i = 0; i < 4; ++i) {
					if (tensor.dims[i] > 1000000) {
						std::cerr << "Warning: Dimension " << i << " is very large: " << tensor.dims[i] << std::endl;
					}
				}

				std::cout << "Parsed tensor: '" << tensor.name << "'" << std::endl;
				std::cout << "  Dimensions: [" << tensor.dims[0] << ", " << tensor.dims[1] << ", " << tensor.dims[2] << ", " << tensor.dims[3] << "]" << std::endl;
				std::cout << "  Type: " << static_cast<int>(tensor.type) << std::endl;
				std::cout << "  Op: " << static_cast<int>(tensor.op) << std::endl;
				std::cout << "  Input names count: " << tensor.input_names.size() << std::endl;
				for (uint64_t i = 0; i < tensor.input_names.size(); ++i) {
					std::cout << "    Input " << i << ": '" << tensor.input_names[i] << "'" << std::endl;
				}
			}

			return tensor;
		} catch (const std::exception& e) {
			std::cerr << "Error parsing tensor: " << e.what() << std::endl;
			throw;
		}
	}

	RT_TM_FORCE_INLINE intermediary_tensor parse_tensor(const std::string& other) {
		return parse_tensor_from_string(other);
	}

	template<bool exceptions, typename value_type> struct debugging_io;

	template<bool exceptions> struct debugging_io<exceptions, struct core_base_creation_data> {
		RT_TM_FORCE_INLINE static void load_and_compare_tensors(const core_base_creation_data& core) {
			const auto& new_string = file_loader<exceptions>{ std::string{ convert_rt_tm_name_to_ggml(core.name) } + ".safetensor" }.operator const std::string&();
			if (new_string.size() > 0) {
				intermediary_tensor new_tensor{ parse_tensor(new_string) };
				intermediary_tensor save_tensor{ core };
				if (new_tensor != save_tensor) {
					std::cout << "Failed on Tensor: " << new_tensor.name << std::endl;
				} else {
					std::cout << "Success on Tensor: " << new_tensor.name << std::endl;
				}
			}
		}
	};
}
