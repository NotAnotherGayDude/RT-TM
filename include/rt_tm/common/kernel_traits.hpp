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

	template<typename... types> struct error_printer_impl;

	template<bool value, typename... value_to_test> struct static_assert_printer {
		static constexpr bool impl{ [] {
			if constexpr (!value) {
				error_printer_impl<value_to_test...>::failure_value;
				return false;
			} else {
				return true;
			}
		}() };
	};

	template<kernel_type kernel, typename... op_types> struct kernel_traits;

	template<typename output, typename input01, typename input02> struct kernel_traits<kernel_type::mul, output, input01, input02> {
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "MUL: Input dimensions[0] must match");
		static_assert(static_assert_printer<(input01::dims[0] == output::dims[0]), kernel_traits, output, input01, input02>::impl, "MUL: Output dimensions[0] must match inputs");
		static_assert(static_assert_printer<(input02::dims[1] == 1 || input01::dims[1] == input02::dims[1]), kernel_traits, output, input01, input02>::impl,
			"MUL: Broadcasting requires input02[1] = 1 or matching dimensions[1]");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02>::impl, "MUL: Output dimensions[1] must match input01");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "MUL: Batch dimensions[2] must match");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "MUL: Batch dimensions[3] must match");
		static constexpr auto input01_dims	  = input01::dims;
		static constexpr auto input02_dims	  = input02::dims;
		static constexpr auto output_dims	  = output::dims;
		using input_type01					  = typename input01::output_type;
		using input_type02					  = typename input02::output_type;
		using output_type					  = typename output::output_type;
		static constexpr bool is_broadcasting		   = (input02_dims[1] == 1 && input01_dims[1] > 1);
		static constexpr size_t total_elements		   = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static constexpr size_t input01_total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t input02_total_elements = input02_dims[0] * input02_dims[1] * input02_dims[2] * input02_dims[3];
		static_assert(static_assert_printer<(total_elements == input01_total_elements), kernel_traits, output, input01, input02>::impl,
			"MUL: Total element count must match between inputs");
		static_assert(static_assert_printer < (total_elements == input02_total_elements) || is_broadcasting, kernel_traits, output, input01, input02 > ::impl,
			"MUL: Total element count must match between input and output");
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::rms_norm, output, input01> {
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01>::impl,
			"RMS_NORM: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01>::impl,
			"RMS_NORM: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01>::impl,
			"RMS_NORM: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01>::impl,
			"RMS_NORM: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename input01::output_type>, "RMS_NORM: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "RMS_NORM: Output type must be valid activation type");
		static constexpr auto input01_dims			  = input01::dims;
		static constexpr auto output_dims			  = output::dims;
		using input_type01							  = typename input01::output_type;
		using output_type							  = typename output::output_type;
		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer<(input_total_elements == output_total_elements), kernel_traits, output, input01>::impl,
			"RMS_NORM: Total element count must match between input and output");
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::silu, output, input01> {
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename input01::output_type>, "SILU: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename output::output_type>, "SILU: Output type must be valid activation type");

		static constexpr auto input01_dims	   = input01::dims;
		static constexpr auto output_dims	   = output::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename output::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

	template<typename output, typename input01, typename input02> struct kernel_traits<kernel_type::softmax, output, input01, input02> {
		static_assert(static_assert_printer<(input01::dims[0] == output::dims[0]), kernel_traits, output, input01, input02>::impl,
			"SOFTMAX: Output dimensions[0] must match input01");
		static_assert(static_assert_printer<(input01::dims[1] == output::dims[1]), kernel_traits, output, input01, input02>::impl,
			"SOFTMAX: Output dimensions[1] must match input01");
		static_assert(static_assert_printer<(input01::dims[2] == output::dims[2]), kernel_traits, output, input01, input02>::impl,
			"SOFTMAX: Output dimensions[2] must match input01");
		static_assert(static_assert_printer<(input01::dims[3] == output::dims[3]), kernel_traits, output, input01, input02>::impl,
			"SOFTMAX: Output dimensions[3] must match input01");
		static_assert(static_assert_printer<(input02::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl,
			"SOFTMAX: Mask dimensions[0] must match scores");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto input02_dims = input02::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using input_type02				   = typename input02::output_type;
		using output_type				   = typename output::output_type;
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::reshape, output, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "RESHAPE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "RESHAPE: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;
		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer<(input_total_elements == output_total_elements), kernel_traits, output, input01>::impl,
			"RESHAPE: Total element count must be preserved");
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::transpose, output, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "TRANSPOSE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "TRANSPOSE: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;
		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer<(input_total_elements == output_total_elements), kernel_traits, output, input01>::impl,
			"TRANSPOSE: Total element count must be preserved");
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::permute, output, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "PERMUTE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "PERMUTE: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;
		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer<(input_total_elements == output_total_elements), kernel_traits, output, input01>::impl,
			"PERMUTE: Total element count must be preserved");
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::cont, output, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "CONT: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "CONT: Output type must be valid tensor type");
		static constexpr auto input01_dims			  = input01::dims;
		static constexpr auto output_dims			  = output::dims;
		using input_type01							  = typename input01::output_type;
		using output_type							  = typename output::output_type;
		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer<(input_total_elements == output_total_elements), kernel_traits, output, input01>::impl,
			"CONT: Total element count must match between input and output");
		static constexpr size_t total_elements = input_total_elements;
	};

	template<typename output, typename input01> struct kernel_traits<kernel_type::view, output, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "VIEW: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename output::output_type>, "VIEW: Output type must be valid tensor type");
		static constexpr auto input01_dims = input01::dims;
		static constexpr auto output_dims  = output::dims;
		using input_type01				   = typename input01::output_type;
		using output_type				   = typename output::output_type;
		static constexpr size_t input_total_elements  = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t output_total_elements = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer<(input_total_elements >= output_total_elements), kernel_traits, output, input01>::impl,
			"VIEW: Output cannot have more elements than input");
	};

	template<typename output, typename input01, typename input02> struct kernel_traits<kernel_type::mul_mat, output, input01, input02> {
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Weight rows must match input vector size");
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[1]), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Output size must match weight columns");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2] || (input01::dims[2] * (input02::dims[2] / input01::dims[2]) == input02::dims[2])), kernel_traits,
						  output, input01, input02>::impl,
			"MUL_MAT: Batch dimension[2] must match or support GQA broadcasting");
		static_assert(static_assert_printer<(output::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Output head count must match attention head count");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "MUL_MAT: Batch dimension[3] must match");
		static_assert(static_assert_printer<(output::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Output head count must match attention head count");
		static_assert(static_assert_printer<(output::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Output batch dimension[3] must match attention dimensions");
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
		static constexpr size_t expected_output_elements = M * (input02_dims[1] == 1 ? 1 : N) * batch_size;
		static constexpr size_t actual_output_elements	 = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
	};

	template<typename output, typename input01, typename input02> struct kernel_traits<kernel_type::get_rows, output, input01, input02> {
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl,
			"GET_ROWS: Output rows must match number of indices");
		static_assert(static_assert_printer<(output::dims[1] == input02::dims[0]), kernel_traits, output, input01, input02>::impl,
			"GET_ROWS: Output sequence length must match input token count");
		static_assert(static_assert_printer<(output::dims[2] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Output dimension[2] must be 1");
		static_assert(static_assert_printer<(output::dims[3] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Output dimension[3] must be 1");
		static_assert(static_assert_printer<(input02::dims[1] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Index tensor dimension[1] must be 1");
		static_assert(static_assert_printer<(input02::dims[2] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Index tensor dimension[2] must be 1");
		static_assert(static_assert_printer<(input02::dims[3] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Index tensor dimension[3] must be 1");
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

	template<typename output, typename input01, typename input02, typename input03> struct kernel_traits<kernel_type::rope, output, input01, input02, input03> {
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Output dimensions must match input tensor");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Sequence length must match");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Number of heads must match");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Batch dimension must match");
		static_assert(static_assert_printer<(input02::dims[0] == input01::dims[1]), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Position count must match sequence length");
		static_assert(static_assert_printer<(input02::dims[1] == 1 && input02::dims[2] == 1 && input02::dims[3] == 1), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Position indices must be 1D");
		static_assert(static_assert_printer<(input03::dims[0] == input01::dims[0] / 2), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Frequency count must be head_dim/2");
		static_assert(static_assert_printer<(input03::dims[1] == 1 && input03::dims[2] == 1 && input03::dims[3] == 1), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Frequencies must be 1D");
		static_assert(static_assert_printer<(input01::dims[0] % 2 == 0), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Head dimension must be even");
		using input_type01						= typename input01::output_type;
		using input_type02						= typename input02::output_type;
		using input_type03						= typename input03::output_type;
		using output_type						= typename output::output_type;
		static constexpr size_t head_dim		= input01::dims[0];
		static constexpr size_t sequence_length = input01::dims[1];
		static constexpr size_t num_heads		= input01::dims[2];
	};

	template<typename output, typename input01, typename input02> struct kernel_traits<kernel_type::add, output, input01, input02> {
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[0] must match");
		static_assert(static_assert_printer<(input01::dims[1] == input02::dims[1]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[1] must match");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[2] must match");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[3] must match");
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl,
			"ADD: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02>::impl,
			"ADD: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01, input02>::impl,
			"ADD: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01, input02>::impl,
			"ADD: Output dimensions[3] must match input dimensions");
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

	template<typename output, typename input01, typename input02> struct kernel_traits<kernel_type::sub, output, input01, input02> {
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[0] must match");
		static_assert(static_assert_printer<(input01::dims[1] == input02::dims[1]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[1] must match");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[2] must match");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[3] must match");
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl,
			"SUB: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02>::impl,
			"SUB: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01, input02>::impl,
			"SUB: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01, input02>::impl,
			"SUB: Output dimensions[3] must match input dimensions");
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
		static_assert(is_valid_tensor_type<typename input01::output_type>, "COPY: Source type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename input02::output_type>, "COPY: Destination type must be valid tensor type");
		static constexpr auto input01_dims		= input01::dims;
		static constexpr auto input02_dims		= input02::dims;
		using input_type01						= typename input01::output_type;
		using input_type02						= typename input02::output_type;
		static constexpr size_t source_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
		static constexpr size_t dest_elements	= input02_dims[0] * input02_dims[1] * input02_dims[2] * input02_dims[3];
		static_assert(static_assert_printer<(source_elements == dest_elements), kernel_traits, input01, input02>::impl,
			"COPY: Source and destination must have same total element count");
		static constexpr size_t total_elements = source_elements;
	};

	template<typename input01> struct kernel_traits<kernel_type::none, input01> {
		static_assert(is_valid_tensor_type<typename input01::output_type>, "NONE: Type must be valid tensor type");
		static constexpr auto input01_dims	   = input01::dims;
		using input_type01					   = typename input01::output_type;
		using output_type					   = typename input01::output_type;
		static constexpr size_t total_elements = input01_dims[0] * input01_dims[1] * input01_dims[2] * input01_dims[3];
	};

}
