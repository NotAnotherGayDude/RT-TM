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
#include <nihilus/common/array.hpp>
#include <latch>

namespace nihilus {

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

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, typename... operand_types> struct kernel_dispatcher_impl;

	template<uint64_t cpu_arch_index, auto op_type, kernel_type krn_type, typename transform_type, typename core_type, typename... operand_types> struct kernel_traits_base;

	template<uint64_t cpu_arch_index, auto op_type, kernel_type krn_type, typename transform_type, single_input core_type, typename output_type, typename input_type01>
	struct kernel_traits_base<cpu_arch_index, op_type, krn_type, transform_type, core_type, output_type, input_type01> {
		using input01									 = typename core_type::input_type01;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::dims;
		static constexpr auto dims02					 = core_type::input_type01::dims;
		static constexpr auto strides01					 = core_type::strides;
		static constexpr auto strides02					 = core_type::input_type01::strides;
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ), kernel_traits_base, core_type, output_type, input_type01>::impl,
			"Sorry, but these output types are not the same");
		static_assert(
			static_assert_printer<( std::is_same_v<input_type01, typename core_type::input_type01::output_type> ), kernel_traits_base, core_type, output_type, input_type01>::impl,
			"Sorry, but these input_type01 types are not the same");
	};

	template<uint64_t cpu_arch_index, auto op_type, kernel_type krn_type, typename transform_type, double_input core_type, typename output_type, typename input_type01,
		typename input_type02>
	struct kernel_traits_base<cpu_arch_index, op_type, krn_type, transform_type, core_type, output_type, input_type01, input_type02> {
		using input01									 = typename core_type::input_type01;
		using input02									 = typename core_type::input_type02;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::dims;
		static constexpr auto dims02					 = core_type::input_type01::dims;
		static constexpr auto dims03					 = core_type::input_type02::dims;
		static constexpr auto strides01					 = core_type::strides;
		static constexpr auto strides02					 = core_type::input_type01::strides;
		static constexpr auto strides03					 = core_type::input_type02::strides;
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static_assert(
			static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ), kernel_traits_base, core_type, output_type, input_type01, input_type02>::impl,
			"Sorry, but these output types are not the same");
		static_assert(static_assert_printer<( std::is_same_v<input_type01, typename core_type::input_type01::output_type> ), kernel_traits_base, core_type, output_type,
						  input_type01, input_type02>::impl,
			"Sorry, but these input_type01 types are not the same");
		static_assert(static_assert_printer<( std::is_same_v<input_type02, typename core_type::input_type02::output_type> ), kernel_traits_base, core_type, output_type,
						  input_type01, input_type02>::impl,
			"Sorry, but these input_type02 types are not the same");
	};

	template<uint64_t cpu_arch_index, auto op_type, kernel_type krn_type, typename transform_type, triple_input core_type, typename output_type, typename input_type01,
		typename input_type02, typename input_type03>
	struct kernel_traits_base<cpu_arch_index, op_type, krn_type, transform_type, core_type, output_type, input_type01, input_type02, input_type03> {
		using input01									 = typename core_type::input_type01;
		using input02									 = typename core_type::input_type02;
		using input03									 = typename core_type::input_type03;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::dims;
		static constexpr auto dims02					 = core_type::input_type01::dims;
		static constexpr auto dims03					 = core_type::input_type02::dims;
		static constexpr auto dims04					 = core_type::input_type03::dims;
		static constexpr auto strides01					 = core_type::strides;
		static constexpr auto strides02					 = core_type::input_type01::strides;
		static constexpr auto strides03					 = core_type::input_type02::strides;
		static constexpr auto strides04					 = core_type::input_type03::strides;
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static constexpr uint64_t input03_total_elements = dims04[0] * dims04[1] * dims04[2] * dims04[3];
		static_assert(
			static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ), kernel_traits_base, core_type, output_type, input_type01, input_type02>::impl,
			"Sorry, but these output types are not the same");
		static_assert(static_assert_printer<( std::is_same_v<input_type01, typename core_type::input_type01::output_type> ), kernel_traits_base, core_type, output_type,
						  input_type01, input_type02, input_type03>::impl,
			"Sorry, but these input_type01 types are not the same");
		static_assert(static_assert_printer<( std::is_same_v<input_type02, typename core_type::input_type02::output_type> ), kernel_traits_base, core_type, output_type,
						  input_type01, input_type02, input_type03>::impl,
			"Sorry, but these input_type02 types are not the same");
		static_assert(static_assert_printer<( std::is_same_v<input_type03, typename core_type::input_type03::output_type> ), kernel_traits_base, core_type, output_type,
						  input_type01, input_type02, input_type03>::impl,
			"Sorry, but these input_type03 types are not the same");
	};

	template<uint64_t cpu_arch_index, auto op_type, kernel_type krn_type, typename transform_type, typename core_type, typename... operand_types> struct kernel_traits;

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, double_input core_type> struct kernel_traits<cpu_arch_index, op_type, kernel_type::mul, transform_type,
		core_type, typename core_type::output_type, typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::mul, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::mul, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "MUL: Input dimensions[0] must match");
		static_assert(static_assert_printer<(input01::dims[0] == output::dims[0]), kernel_traits, output, input01, input02>::impl, "MUL: Output dimensions[0] must match inputs");
		static_assert(static_assert_printer<(input02::dims[1] == 1 || input01::dims[1] == input02::dims[1]), kernel_traits, output, input01, input02>::impl,
			"MUL: Broadcasting requires input02[1] = 1 or matching dimensions[1]");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02>::impl, "MUL: Output dimensions[1] must match input01");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "MUL: Batch dimensions[2] must match");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "MUL: Batch dimensions[3] must match");
		static constexpr bool is_broadcasting = (input02::dims[1] == 1 && input01::dims[1] > 1);
		static_assert(static_assert_printer<(base_type::total_elements == base_type::input01_total_elements ), kernel_traits, output, input01, input02>::impl,
			"MUL: Total element count must match between inputs");
		static_assert(static_assert_printer < (base_type::total_elements == base_type::input02_total_elements ) || is_broadcasting, kernel_traits, output, input01, input02 > ::impl,
			"MUL: Total element count must match between input and output");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::rms_norm, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::rms_norm, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::rms_norm, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01>::impl, "RMS_NORM: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01>::impl, "RMS_NORM: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01>::impl, "RMS_NORM: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01>::impl, "RMS_NORM: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename core_type::input_type01::output_type>, "RMS_NORM: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename core_type::output_type>, "RMS_NORM: Output type must be valid activation type");
		static_assert(static_assert_printer<(base_type::input01_total_elements  == base_type::total_elements), kernel_traits, output, input01>::impl,
			"RMS_NORM: Total element count must match between input and output");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::silu, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::silu, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::silu, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01>::impl, "SILU: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_activation_type<typename core_type::input_type01::output_type>, "SILU: Input type must be valid activation type");
		static_assert(is_valid_activation_type<typename core_type::output_type>, "SILU: Output type must be valid activation type");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, double_input core_type> struct kernel_traits<cpu_arch_index, op_type, kernel_type::softmax,
		transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::softmax, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::softmax, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static_assert(static_assert_printer<(input01::dims[0] == output::dims[0]), kernel_traits, output, input01, input02>::impl, "SOFTMAX: Output dimensions[0] must match input01");
		static_assert(static_assert_printer<(input01::dims[1] == output::dims[1]), kernel_traits, output, input01, input02>::impl, "SOFTMAX: Output dimensions[1] must match input01");
		static_assert(static_assert_printer<(input01::dims[2] == output::dims[2]), kernel_traits, output, input01, input02>::impl, "SOFTMAX: Output dimensions[2] must match input01");
		static_assert(static_assert_printer<(input01::dims[3] == output::dims[3]), kernel_traits, output, input01, input02>::impl, "SOFTMAX: Output dimensions[3] must match input01");
		static_assert(static_assert_printer<(input02::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl, "SOFTMAX: Mask dimensions[0] must match scores");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::reshape, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::reshape, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::reshape, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "RESHAPE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "RESHAPE: Output type must be valid tensor type");
		static_assert(static_assert_printer<(base_type::input01_total_elements  == base_type::total_elements), kernel_traits, output, input01>::impl, "RESHAPE: Total element count must be preserved");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::transpose, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::transpose, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::transpose, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "TRANSPOSE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "TRANSPOSE: Output type must be valid tensor type");
		static_assert(static_assert_printer<(base_type::input01_total_elements  == base_type::total_elements), kernel_traits, output, input01>::impl, "TRANSPOSE: Total element count must be preserved");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::permute, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::permute, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::permute, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "PERMUTE: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "PERMUTE: Output type must be valid tensor type");
		static_assert(static_assert_printer<(base_type::input01_total_elements  == base_type::total_elements), kernel_traits, output, input01>::impl, "PERMUTE: Total element count must be preserved");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::cont, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::cont, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::cont, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "CONT: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "CONT: Output type must be valid tensor type");
		static_assert(static_assert_printer<(base_type::input01_total_elements  == base_type::total_elements), kernel_traits, output, input01>::impl,
			"CONT: Total element count must match between input and output");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::view, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::view, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::view, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "VIEW: Input type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "VIEW: Output type must be valid tensor type");
		static_assert(static_assert_printer<(base_type::input01_total_elements  >= base_type::total_elements), kernel_traits, output, input01>::impl, "VIEW: Output cannot have more elements than input");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, double_input core_type> struct kernel_traits<cpu_arch_index, op_type, kernel_type::mul_mat,
		transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::mul_mat, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::mul_mat, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "MUL_MAT: Weight rows must match input vector size");
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[1]), kernel_traits, output, input01, input02>::impl, "MUL_MAT: Output size must match weight columns");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2] || (input01::dims[2] * (input02::dims[2] / input01::dims[2]) == input02::dims[2])), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Batch dimension[2] must match or support GQA broadcasting");
		static_assert(static_assert_printer<(output::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "MUL_MAT: Output head count must match attention head count");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "MUL_MAT: Batch dimension[3] must match");
		static_assert(static_assert_printer<(output::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl,
			"MUL_MAT: Output batch dimension[3] must match attention dimensions");
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "MUL_MAT: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::input_type02::output_type>, "MUL_MAT: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "MUL_MAT: Output type must be valid tensor type");
		static constexpr uint64_t M						   = input01::dims[0];
		static constexpr uint64_t K						   = input01::dims[1];
		static constexpr uint64_t N						   = input02::dims[1];
		static constexpr uint64_t batch_size			   = input01::dims[2] * input01::dims[3];
		static constexpr uint64_t expected_output_elements = M * (input02::dims[1] == 1 ? 1 : N) * batch_size;
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, double_input core_type> struct kernel_traits<cpu_arch_index, op_type, kernel_type::get_rows,
		transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::get_rows, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::get_rows, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Output rows must match number of indices");
		static_assert(static_assert_printer<(output::dims[1] == input02::dims[0]), kernel_traits, output, input01, input02>::impl,
			"GET_ROWS: Output sequence length must match input token count");
		static_assert(static_assert_printer<(output::dims[2] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Output dimension[2] must be 1");
		static_assert(static_assert_printer<(output::dims[3] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Output dimension[3] must be 1");
		static_assert(static_assert_printer<(input02::dims[1] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Index tensor dimension[1] must be 1");
		static_assert(static_assert_printer<(input02::dims[2] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Index tensor dimension[2] must be 1");
		static_assert(static_assert_printer<(input02::dims[3] == 1), kernel_traits, output, input01, input02>::impl, "GET_ROWS: Index tensor dimension[3] must be 1");
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "GET_ROWS: Embedding matrix type must be valid tensor type");
		static_assert(is_integral_type<typename core_type::input_type02::output_type>, "GET_ROWS: Index type must be integer type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "GET_ROWS: Output type must be valid tensor type");
		static constexpr uint64_t vocab_size	  = input01::dims[0];
		static constexpr uint64_t embedding_dim	  = input01::dims[1];
		static constexpr uint64_t sequence_length = input02::dims[0];
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, triple_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::rope, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
		typename core_type::input_type02::output_type, typename core_type::input_type03::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::rope, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type, typename core_type::input_type03::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::rope, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type, typename core_type::input_type03::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using input03	= base_type::input03;
		using output	= base_type::output;
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Output dimensions must match input tensor");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Sequence length must match");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Number of heads must match");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Batch dimension must match");
		static_assert(static_assert_printer<(input02::dims[0] == input01::dims[1]), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Position count must match sequence length");
		static_assert(static_assert_printer<(input02::dims[1] == 1 && input02::dims[2] == 1 && input02::dims[3] == 1), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Position indices must be 1D");
		static_assert(static_assert_printer<(input03::dims[0] == input01::dims[0] / 2), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Frequency count must be head_dim/2");
		static_assert(static_assert_printer<(input03::dims[1] == 1 && input03::dims[2] == 1 && input03::dims[3] == 1), kernel_traits, output, input01, input02, input03>::impl,
			"ROPE: Frequencies must be 1D");
		static_assert(static_assert_printer<(input01::dims[0] % 2 == 0), kernel_traits, output, input01, input02, input03>::impl, "ROPE: Head dimension must be even");
		static constexpr uint64_t head_dim		  = input01::dims[0];
		static constexpr uint64_t sequence_length = input01::dims[1];
		static constexpr uint64_t num_heads		  = input01::dims[2];
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, double_input core_type> struct kernel_traits<cpu_arch_index, op_type, kernel_type::add, transform_type,
		core_type, typename core_type::output_type, typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::add, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::add, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[0] must match");
		static_assert(static_assert_printer<(input01::dims[1] == input02::dims[1]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[1] must match");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[2] must match");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "ADD: Input dimensions[3] must match");
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl, "ADD: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02>::impl, "ADD: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01, input02>::impl, "ADD: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01, input02>::impl, "ADD: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "ADD: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::input_type02::output_type>, "ADD: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "ADD: Output type must be valid tensor type");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, double_input core_type> struct kernel_traits<cpu_arch_index, op_type, kernel_type::sub, transform_type,
		core_type, typename core_type::output_type, typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::sub, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type, typename core_type::input_type02::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::sub, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static_assert(static_assert_printer<(input01::dims[0] == input02::dims[0]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[0] must match");
		static_assert(static_assert_printer<(input01::dims[1] == input02::dims[1]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[1] must match");
		static_assert(static_assert_printer<(input01::dims[2] == input02::dims[2]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[2] must match");
		static_assert(static_assert_printer<(input01::dims[3] == input02::dims[3]), kernel_traits, output, input01, input02>::impl, "SUB: Input dimensions[3] must match");
		static_assert(static_assert_printer<(output::dims[0] == input01::dims[0]), kernel_traits, output, input01, input02>::impl, "SUB: Output dimensions[0] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[1] == input01::dims[1]), kernel_traits, output, input01, input02>::impl, "SUB: Output dimensions[1] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[2] == input01::dims[2]), kernel_traits, output, input01, input02>::impl, "SUB: Output dimensions[2] must match input dimensions");
		static_assert(static_assert_printer<(output::dims[3] == input01::dims[3]), kernel_traits, output, input01, input02>::impl, "SUB: Output dimensions[3] must match input dimensions");
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "SUB: Input1 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::input_type02::output_type>, "SUB: Input2 type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "SUB: Output type must be valid tensor type");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::copy, transform_type, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::copy, transform_type, core_type, typename core_type::output_type,
			  typename core_type::input_type01::output_type> {
		using base_type = kernel_traits_base<cpu_arch_index, op_type, kernel_type::copy, transform_type, core_type, typename core_type::output_type,
			typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "COPY: Source type must be valid tensor type");
		static_assert(is_valid_tensor_type<typename core_type::output_type>, "COPY: Destination type must be valid tensor type");
		static constexpr uint64_t source_elements = output::dims[0] * output::dims[1] * output::dims[2] * output::dims[3];
		static constexpr uint64_t dest_elements	  = input01::dims[0] * input01::dims[1] * input01::dims[2] * input01::dims[3];
		static_assert(static_assert_printer<(source_elements == dest_elements), kernel_traits, input01>::impl, "COPY: Source and destination must have same total element count");
	};

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, single_input core_type>
	struct kernel_traits<cpu_arch_index, op_type, kernel_type::none, transform_type, core_type, typename core_type::input_type01::output_type>
		: public kernel_traits_base<cpu_arch_index, op_type, kernel_type::none, transform_type, core_type, typename core_type::input_type01::output_type> {
		static_assert(is_valid_tensor_type<typename core_type::input_type01::output_type>, "NONE: Type must be valid tensor type");
	};

}
