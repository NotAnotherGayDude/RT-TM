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
#include <rt_tm/common/core_base.hpp>
#include <rt_tm/common/model_arch_traits.hpp>
#include <rt_tm/common/monolithic_dispatcher.hpp>

namespace rt_tm {

	template<typename model_arch_traits, op_type type> struct produce_op;

	template<typename model_arch_traits> struct produce_op<model_arch_traits, op_type::get_rows> {
		template<typename name_type>
		RT_TM_FORCE_INLINE static core_base_creation_data impl(core_base_creation_data& lhs, core_base_creation_data& rhs, size_t op_id, name_type name, size_t layer_index) {
			core_base_creation_data result{};
			result.name			  = model_arch_traits::op_names[static_cast<size_t>(name)][layer_index];
			result.allocated_dims = { { lhs.allocated_dims[0], rhs.allocated_dims[0], 1, 1 } };
			result.allocated_dims = { { lhs.allocated_dims[0], rhs.allocated_dims[0], 1, 1 } };
			result.depth		  = std::max(lhs.depth, rhs.depth) + 1;
			result.blocking		  = result.depth == 0 ? false : true;
			result.input_ops	  = { { &lhs, &rhs } };
			result.data_type_val  = data_type::float_32;
			result.type			  = op_type::get_rows;
			result.op_id		  = op_id;
			data_type types[3]{ lhs.data_type_val, rhs.data_type_val, result.data_type_val };
			result.comparison_index = get_comparison_value<op_entity<op_type::get_rows>>::impl(types);
			result.allocate_memory	= true;
			lhs.dependent_ops.emplace_back(result.op_id);
			rhs.dependent_ops.emplace_back(result.op_id);
			return result;
		}
	};

	template<typename model_arch_traits> struct produce_op<model_arch_traits, op_type::rms_norm> {
		template<typename name_type>
		RT_TM_FORCE_INLINE static core_base_creation_data impl(core_base_creation_data& lhs, size_t op_id, name_type name, size_t layer_index, double rms_norm_epsilon) {
			core_base_creation_data result{};
			result.type			  = op_type::rms_norm;
			result.op_id		  = op_id;
			result.name			  = model_arch_traits::op_names[static_cast<size_t>(name)][layer_index];
			result.allocated_dims = lhs.allocated_dims;
			result.allocated_dims = lhs.allocated_dims;
			result.data_type_val  = lhs.data_type_val;
			result.input_ops	  = { { &lhs } };
			result.depth		  = lhs.depth + 1;
			result.blocking		  = result.depth == 0 ? false : true;
			data_type types[2]{ lhs.data_type_val, result.data_type_val };
			result.comparison_index = get_comparison_value<op_entity<op_type::rms_norm>>::impl(types);
			result.set_value<rms_norm_aux_params::rms_norm_epsilon>(rms_norm_epsilon);
			result.allocate_memory = true;
			lhs.dependent_ops.emplace_back(result.op_id);
			return result;
		}
	};

	template<typename model_arch_traits> struct produce_op<model_arch_traits, op_type::mul> {
		template<typename name_type>
		RT_TM_FORCE_INLINE static core_base_creation_data impl(core_base_creation_data& lhs, core_base_creation_data& rhs, size_t op_id, name_type name, size_t layer_index) {
			core_base_creation_data result{};
			result.type			   = op_type::mul;
			result.op_id		   = op_id;
			result.name			   = model_arch_traits::op_names[static_cast<size_t>(name)][layer_index];
			result.allocated_dims  = { { std::max(lhs.allocated_dims[0], rhs.allocated_dims[0]), std::max(lhs.allocated_dims[1], rhs.allocated_dims[1]),
				 std::max(lhs.allocated_dims[2], rhs.allocated_dims[2]), std::max(lhs.allocated_dims[3], rhs.allocated_dims[3]) } };
			result.data_type_val   = lhs.data_type_val;
			result.input_ops	   = { { &lhs, &rhs } };
			result.depth		   = std::max(lhs.depth, rhs.depth) + 1;
			data_type types[3]{ lhs.data_type_val, rhs.data_type_val, result.data_type_val };
			result.comparison_index = get_comparison_value<op_entity<op_type::mul>>::impl(types);
			result.allocate_memory = true;
			rhs.dependent_ops.emplace_back(result.op_id);
			lhs.dependent_ops.emplace_back(result.op_id);
			return result;
		}
	};

	template<typename model_arch_traits> struct produce_op<model_arch_traits, op_type::mul_mat> {
		template<typename name_type>
		RT_TM_FORCE_INLINE static core_base_creation_data impl(core_base_creation_data& lhs, core_base_creation_data& rhs, size_t op_id, name_type name, size_t layer_index) {
			core_base_creation_data result{};
			result.type				= op_type::mul_mat;
			result.op_id			= op_id;
			result.name				= model_arch_traits::op_names[static_cast<size_t>(name)][layer_index];
			const size_t batch_dim1 = std::max(lhs.allocated_dims[2], rhs.allocated_dims[2]);
			const size_t batch_dim2 = std::max(lhs.allocated_dims[3], rhs.allocated_dims[3]);
			const size_t batch_size = lhs.allocated_dims[0];
			const size_t lhs_inner	= lhs.allocated_dims[1];
			const size_t rhs_inner	= rhs.allocated_dims[1];
			result.allocated_dims	= { { batch_size, lhs_inner, batch_dim1, batch_dim2 } };
			result.data_type_val	= lhs.data_type_val;
			result.input_ops		= { { &lhs, &rhs } };
			result.depth			= std::max(lhs.depth, rhs.depth) + 1;
			result.allocate_memory	= true;
			result.blocking			= true;
			data_type types[3]{ lhs.data_type_val, rhs.data_type_val, result.data_type_val };
			result.comparison_index = get_comparison_value<op_entity<op_type::mul_mat>>::impl(types);
			rhs.dependent_ops.emplace_back(result.op_id);
			lhs.dependent_ops.emplace_back(result.op_id);
			return result;
		}
	};

	template<typename model_arch_traits> struct produce_op<model_arch_traits, op_type::reshape> {
		template<typename name_type> RT_TM_FORCE_INLINE static core_base_creation_data impl(core_base_creation_data& lhs, size_t op_id, name_type name, size_t layer_index,
			size_t new_dim0 = 0, size_t new_dim1 = 0, size_t new_dim2 = 0, size_t new_dim3 = 0) {
			core_base_creation_data result{};
			result.type			   = op_type::reshape;
			result.op_id		   = op_id;
			result.name			   = model_arch_traits::op_names[static_cast<size_t>(name)][layer_index];
			size_t dim00		   = new_dim0 == 0 ? lhs.allocated_dims[0] : new_dim0;
			size_t dim01		   = new_dim1 == 0 ? lhs.allocated_dims[1] : new_dim1;
			size_t dim02		   = new_dim2 == 0 ? lhs.allocated_dims[2] : new_dim2;
			size_t dim03		   = new_dim3 == 0 ? lhs.allocated_dims[3] : new_dim3;
			result.allocated_dims  = { { dim00, dim01, dim02, dim03 } };
			result.data_type_val   = lhs.data_type_val;
			result.input_ops	   = { { &lhs } };
			result.depth		   = lhs.depth + 1;
			result.allocate_memory = false;
			result.blocking		   = false;
			data_type types[2]{ lhs.data_type_val, result.data_type_val };
			result.comparison_index = get_comparison_value<op_entity<op_type::reshape>>::impl(types);
			lhs.dependent_ops.emplace_back(result.op_id);
			return result;
		}
	};

	template<typename model_arch_traits> struct produce_op<model_arch_traits, op_type::rope> {
		template<typename name_type> RT_TM_FORCE_INLINE static core_base_creation_data impl(core_base_creation_data& a, core_base_creation_data& b, core_base_creation_data& c,
			size_t op_id, name_type name, size_t layer_index, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor,
			float beta_fast, float beta_slow, bool inplace) {
			core_base_creation_data result{};
			result.type	 = op_type::rope;
			result.op_id = op_id;

			a.dependent_ops.emplace_back(result.op_id);
			b.dependent_ops.emplace_back(result.op_id);
			c.dependent_ops.emplace_back(result.op_id);
			data_type types[4]{ a.data_type_val, b.data_type_val, c.data_type_val, result.data_type_val };
			result.comparison_index = get_comparison_value<op_entity<op_type::rope>>::impl(types);
			return result;
		}
	};

}