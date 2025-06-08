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
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == input02_dims.size(), "MUL: Total element count must match between inputs");
		static_assert(total_elements == output_dims.size(), "MUL: Total element count must match between input and output");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::mul_mat, input01, input02, output> {
		static_assert(input01::dims[1] == input02::dims[0], "MUL_MAT: Input1 columns must match Input2 rows");
		static_assert(output::dims[0] == input01::dims[0], "MUL_MAT: Output rows must match Input1 rows");
		static_assert(output::dims[1] == input02::dims[1], "MUL_MAT: Output columns must match Input2 columns");
		static_assert(output::dims[2] == 1 && output::dims[3] == 1, "MUL_MAT: Output must be 2D matrix");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "MUL_MAT: Input1 type must be valid tensor type");
		static_assert(is_valid_weight_type<typename input02::output_type>, "MUL_MAT: Input2 type must be valid weight type");
		static_assert(is_valid_activation_type<typename output::output_type>, "MUL_MAT: Output type must be valid activation type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
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
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == input02_dims.size(), "ADD: Total element count must match between inputs");
		static_assert(total_elements == output_dims.size(), "ADD: Total element count must match between input and output");
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
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == input02_dims.size(), "SUB: Total element count must match between inputs");
		static_assert(total_elements == output_dims.size(), "SUB: Total element count must match between input and output");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::get_rows, input01, input02, output> {
		static_assert(input01::dims[1] == output::dims[1], "GET_ROWS: Matrix columns must match output columns");
		static_assert(input02::dims[1] == 1 && input02::dims[2] == 1 && input02::dims[3] == 1, "GET_ROWS: Indices must be 1D");
		static_assert(output::dims[0] == input02::dims[0], "GET_ROWS: Output rows must match number of indices");
		static_assert(is_valid_weight_type<typename input01::output_type>, "GET_ROWS: Matrix type must be valid weight type");
		static_assert(is_integral_type<typename input02::output_type>, "GET_ROWS: Index type must be integral");
		static_assert(is_valid_activation_type<typename output::output_type>, "GET_ROWS: Output type must be valid activation type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::view, input01, input02, output> {
		static constexpr size_t input_elements	= input01::dims.size();
		static constexpr size_t output_elements = output::dims.size();
		static_assert(input_elements == output_elements, "VIEW: Total element count must remain the same");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "VIEW: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "VIEW: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::copy, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "COPY: Input/output dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "COPY: Input/output dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[2], "COPY: Input/output dimensions[2] must match");
		static_assert(input01::dims[3] == output::dims[3], "COPY: Input/output dimensions[3] must match");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "COPY: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "COPY: Output type must be valid tensor type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == output_dims.size(), "COPY: Total element count must match");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::softmax, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "SOFTMAX: Input/output dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "SOFTMAX: Input/output dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[2], "SOFTMAX: Input/output dimensions[2] must match");
		static_assert(input01::dims[3] == output::dims[3], "SOFTMAX: Input/output dimensions[3] must match");
		static_assert(is_valid_activation_type<typename input01::output_type>, "SOFTMAX: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "SOFTMAX: Output type must be valid activation type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == output_dims.size(), "SOFTMAX: Total element count must match");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::rms_norm, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "RMS_NORM: Input/output dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "RMS_NORM: Input/output dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[2], "RMS_NORM: Input/output dimensions[2] must match");
		static_assert(input01::dims[3] == output::dims[3], "RMS_NORM: Input/output dimensions[3] must match");
		static_assert(is_valid_activation_type<typename input01::output_type>, "RMS_NORM: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "RMS_NORM: Output type must be valid activation type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == output_dims.size(), "RMS_NORM: Total element count must match");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::reshape, input01, input02, output> {
		static constexpr size_t input_elements	= input01::dims.size();
		static constexpr size_t output_elements = output::dims.size();
		static_assert(input_elements == output_elements, "RESHAPE: Total element count must remain the same");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "RESHAPE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "RESHAPE: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::rope, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "ROPE: Input/output dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "ROPE: Input/output dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[2], "ROPE: Input/output dimensions[2] must match");
		static_assert(input01::dims[3] == output::dims[3], "ROPE: Input/output dimensions[3] must match");
		static_assert(is_valid_activation_type<typename input01::output_type>, "ROPE: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "ROPE: Output type must be valid activation type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == output_dims.size(), "ROPE: Total element count must match");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::transpose, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "TRANSPOSE: Dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "TRANSPOSE: Dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[3], "TRANSPOSE: Input dim[2] must match output dim[3]");
		static_assert(input01::dims[3] == output::dims[2], "TRANSPOSE: Input dim[3] must match output dim[2]");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "TRANSPOSE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "TRANSPOSE: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::permute, input01, input02, output> {
		static constexpr size_t input_elements	= input01::dims.size();
		static constexpr size_t output_elements = output::dims.size();
		static_assert(input_elements == output_elements, "PERMUTE: Total element count must remain the same");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "PERMUTE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "PERMUTE: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::cont, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "CONT: Input/output dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "CONT: Input/output dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[2], "CONT: Input/output dimensions[2] must match");
		static_assert(input01::dims[3] == output::dims[3], "CONT: Input/output dimensions[3] must match");
		static_assert(is_valid_tensor_type<typename input01::output_type>, "CONT: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "CONT: Output type must be valid tensor type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == output_dims.size(), "CONT: Total element count must match");
	};

	template<typename input01, typename input02, typename output> struct kernel_traits<kernel_type::silu, input01, input02, output> {
		static_assert(input01::dims[0] == output::dims[0], "SILU: Input/output dimensions[0] must match");
		static_assert(input01::dims[1] == output::dims[1], "SILU: Input/output dimensions[1] must match");
		static_assert(input01::dims[2] == output::dims[2], "SILU: Input/output dimensions[2] must match");
		static_assert(input01::dims[3] == output::dims[3], "SILU: Input/output dimensions[3] must match");
		static_assert(is_valid_activation_type<typename input01::output_type>, "SILU: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "SILU: Output type must be valid activation type");
		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto input02_dims	   = input02::dims;
		static constexpr auto output_dims	   = output::dims;
		static constexpr size_t total_elements = input01_dims.size();
		static_assert(total_elements == output_dims.size(), "SILU: Total element count must match");
	};

}
