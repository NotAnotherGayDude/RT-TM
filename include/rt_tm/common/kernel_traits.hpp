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

#include <rt_tm/common/common.hpp>
#include <rt_tm/common/array.hpp>
#include <latch>

namespace rt_tm {

	template<kernel_type kernel, typename... op_types> struct kernel_traits;

	template<typename T>
	concept is_arithmetic_type = std::is_arithmetic_v<T>;

	template<typename T>
	concept is_quantized_type = requires {
		sizeof(T);
		!std::is_arithmetic_v<T>;
	};

	template<typename T>
	concept is_fp_type = std::is_floating_point_v<T>;

	template<typename T>
	concept is_integral_type = std::is_integral_v<T>;

	template<typename T>
	concept is_valid_tensor_type = is_arithmetic_type<T> || is_quantized_type<T>;

	template<typename T>
	concept is_valid_weight_type = is_arithmetic_type<T> || is_quantized_type<T>;

	template<typename T>
	concept is_valid_activation_type = is_fp_type<T> || is_quantized_type<T>;

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::mul, input01, input02, output> {
		static_assert(input01::dims[0] == input02::dims[0], "MUL: Input dimensions[0] must match");
		static_assert(input01::dims[1] == input02::dims[1], "MUL: Input dimensions[1] must match");
		static_assert(input01::dims[2] == input02::dims[2], "MUL: Input dimensions[2] must match");
		static_assert(input01::dims[3] == input02::dims[3], "MUL: Input dimensions[3] must match");
		static_assert(output::dims[0] == input01::dims[0], "MUL: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "MUL: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "MUL: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "MUL: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "MUL: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename input02::output_type>, "MUL: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "MUL: Output type must be valid tensor type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = input01::output_type;
		using input_type02					   = input02::output_type;
		using output_type					   = output::output_type;
		static constexpr size_t total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(total_elements == input02_dims.size(), "MUL: Total element count must match between inputs");
		static_assert(total_elements == output_dims.size(), "MUL: Total element count must match between input and output");
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::rms_norm, input01, output> {
		static_assert(output::dims[0] == input01::dims[0], "RMS_NORM: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "RMS_NORM: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "RMS_NORM: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "RMS_NORM: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename input01::output_type>, "RMS_NORM: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "RMS_NORM: Output type must be valid activation type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static_assert(total_elements == (output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3]), "RMS_NORM: Total element count must match between input and output");
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::silu, input01, output> {
		static_assert(output::dims[0] == input01::dims[0], "SILU: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "SILU: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "SILU: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "SILU: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename input01::output_type>, "SILU: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "SILU: Output type must be valid activation type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::softmax, input01, output> {
		static_assert(output::dims[0] == input01::dims[0], "SOFTMAX: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "SOFTMAX: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "SOFTMAX: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "SOFTMAX: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename input01::output_type>, "SOFTMAX: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "SOFTMAX: Output type must be valid activation type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::reshape, input01, output> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "RESHAPE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "RESHAPE: Output type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;

		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(input_total_elements == output_total_elements, "RESHAPE: Total element count must be preserved");
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::transpose, input01, output> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "TRANSPOSE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "TRANSPOSE: Output type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;

		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(input_total_elements == output_total_elements, "TRANSPOSE: Total element count must be preserved");
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::permute, input01, output> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "PERMUTE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "PERMUTE: Output type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;

		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(input_total_elements == output_total_elements, "PERMUTE: Total element count must be preserved");
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::cont, input01, output> {
		static_assert(output::dims[0] == input01::dims[0], "CONT: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "CONT: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "CONT: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "CONT: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "CONT: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "CONT: Output type must be valid tensor type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

	template<typename input01, typename output> struct kernel_traits<kernel_type::view, input01, output> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "VIEW: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "VIEW: Output type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;

		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(input_total_elements >= output_total_elements, "VIEW: Output cannot have more elements than input");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::mul_mat, input01, input02, output> {
		static_assert(input01::dims[1] == input02::dims[0], "MUL_MAT: Input1 columns must match Input2 rows");
		static_assert(output::dims[0] == input01::dims[0], "MUL_MAT: Output rows must match Input1 rows");
		static_assert(output::dims[1] == input02::dims[1], "MUL_MAT: Output columns must match Input2 columns");
		static_assert(input01::dims[2] == input02::dims[2], "MUL_MAT: Batch dimension[2] must match");
		static_assert(input01::dims[3] == input02::dims[3], "MUL_MAT: Batch dimension[3] must match");
		static_assert(output::dims[2] == input01::dims[2], "MUL_MAT: Output batch dimension[2] must match inputs");
		static_assert(output::dims[3] == input01::dims[3], "MUL_MAT: Output batch dimension[3] must match inputs");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "MUL_MAT: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename input02::output_type>, "MUL_MAT: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "MUL_MAT: Output type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using input_type02				   = typename input02::output_type;
		using output_type				   = typename output::output_type;

		static constexpr size_t M		   = input01_dims[0];
		static constexpr size_t K		   = input01_dims[1];
		static constexpr size_t N		   = input02_dims[1];
		static constexpr size_t batch_size = input01_dims[2] * input01_dims[3];
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::get_rows, input01, input02, output> {
		static_assert(output::dims[0] == input02::dims[0], "GET_ROWS: Output rows must match number of indices");
		static_assert(output::dims[1] == input01::dims[1], "GET_ROWS: Output columns must match embedding dimension");
		static_assert(output::dims[2] == 1, "GET_ROWS: Output dimension[2] must be 1");
		static_assert(output::dims[3] == 1, "GET_ROWS: Output dimension[3] must be 1");
		static_assert(input02::dims[1] == 1, "GET_ROWS: Index tensor dimension[1] must be 1");
		static_assert(input02::dims[2] == 1, "GET_ROWS: Index tensor dimension[2] must be 1");
		static_assert(input02::dims[3] == 1, "GET_ROWS: Index tensor dimension[3] must be 1");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "GET_ROWS: Embedding matrix type must be valid tensor type");
		static_assert(is_integral_type<typename input02::output_type>, "GET_ROWS: Index type must be integer type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "GET_ROWS: Output type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using input_type02				   = typename input02::output_type;
		using output_type				   = typename output::output_type;

		static constexpr size_t vocab_size		= input01_dims[0];
		static constexpr size_t embedding_dim	= input01_dims[1];
		static constexpr size_t sequence_length = input02_dims[0];
	};

	template<typename input01, typename input02, typename input03, typename output> struct kernel_traits<kernel_type::rope, input01, input02, input03, output> {
		static_assert(output::dims[0] == input01::dims[0], "ROPE: Output dimensions must match input tensor");
		static_assert(output::dims[1] == input01::dims[1], "ROPE: Sequence length must match");
		static_assert(output::dims[2] == input01::dims[2], "ROPE: Number of heads must match");
		static_assert(output::dims[3] == input01::dims[3], "ROPE: Batch dimension must match");

		static_assert(input02::dims[0] == input01::dims[1], "ROPE: Position count must match sequence length");
		static_assert(input02::dims[1] == 1 && input02::dims[2] == 1 && input02::dims[3] == 1, "ROPE: Position indices must be 1D");

		static_assert(input03::dims[0] == input01::dims[0] / 2, "ROPE: Frequency count must be head_dim/2");
		static_assert(input03::dims[1] == 1 && input03::dims[2] == 1 && input03::dims[3] == 1, "ROPE: Frequencies must be 1D");

		static_assert(input01::dims[0] % 2 == 0, "ROPE: Head dimension must be even");

		using input_type01 = typename input01::output_type;
		using input_type02 = typename input02::output_type;
		using input_type03 = typename input03::output_type;
		using output_type  = typename output::output_type;

		static constexpr size_t head_dim		= input01::dims[0];
		static constexpr size_t sequence_length = input01::dims[1];
		static constexpr size_t num_heads		= input01::dims[2];
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::add, input01, input02, output> {
		static_assert(input01::dims[0] == input02::dims[0], "ADD: Input dimensions[0] must match");
		static_assert(input01::dims[1] == input02::dims[1], "ADD: Input dimensions[1] must match");
		static_assert(input01::dims[2] == input02::dims[2], "ADD: Input dimensions[2] must match");
		static_assert(input01::dims[3] == input02::dims[3], "ADD: Input dimensions[3] must match");
		static_assert(output::dims[0] == input01::dims[0], "ADD: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "ADD: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "ADD: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "ADD: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "ADD: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename input02::output_type>, "ADD: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "ADD: Output type must be valid tensor type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using input_type02					   = typename input02::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::sub, input01, input02, output> {
		static_assert(input01::dims[0] == input02::dims[0], "SUB: Input dimensions[0] must match");
		static_assert(input01::dims[1] == input02::dims[1], "SUB: Input dimensions[1] must match");
		static_assert(input01::dims[2] == input02::dims[2], "SUB: Input dimensions[2] must match");
		static_assert(input01::dims[3] == input02::dims[3], "SUB: Input dimensions[3] must match");
		static_assert(output::dims[0] == input01::dims[0], "SUB: Output dimensions[0] must match input dimensions");
		static_assert(output::dims[1] == input01::dims[1], "SUB: Output dimensions[1] must match input dimensions");
		static_assert(output::dims[2] == input01::dims[2], "SUB: Output dimensions[2] must match input dimensions");
		static_assert(output::dims[3] == input01::dims[3], "SUB: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "SUB: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename input02::output_type>, "SUB: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "SUB: Output type must be valid tensor type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using input_type02					   = typename input02::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

	template<typename input01, typename input02> struct kernel_traits<kernel_type::copy, input01, input02> {
		static_assert(input01::dims[0] <= input02::dims[0], "COPY: Source dimensions[0] must fit in destination");
		static_assert(input01::dims[1] <= input02::dims[1], "COPY: Source dimensions[1] must fit in destination");
		static_assert(input01::dims[2] <= input02::dims[2], "COPY: Source dimensions[2] must fit in destination");
		static_assert(input01::dims[3] <= input02::dims[3], "COPY: Source dimensions[3] must fit in destination");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "COPY: Source type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename input02::output_type>, "COPY: Destination type must be valid tensor type");

		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		using input_type01				   = typename input01::output_type;
		using input_type02				   = typename input02::output_type;

		static constexpr size_t source_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t dest_elements	= input02_dims[0] * input02_dims[1] * input02_dims[2] * input02_dims[3];
		static_assert(source_elements <= dest_elements, "COPY: Source data must fit in destination");
	};

	template<typename input01> struct kernel_traits<kernel_type::none, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "NONE: Type must be valid tensor type");

		static constexpr auto input01_dims	   = input01::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename input01::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

}
