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
#include <rt_tm/common/param_api.hpp>
#include <rt_tm/common/string_view.hpp>

namespace rt_tm {

	struct core_base_creation_data : public param_api<core_base_creation_data> {
		std::vector<core_base_creation_data*> input_ops{};
		array<size_t, 4> allocated_dims{ { 1, 1, 1, 1 } };
		std::vector<size_t> dependent_ops{};
		std::vector<uint8_t> aux_params{};
		size_t comparison_index{};
		data_type data_type_val{};
		bool allocate_memory{};
		mutable void* data{};
		const char* name{};
		bool blocking{};
		size_t op_id{};
		size_t depth{};
		op_type type{};

		RT_TM_FORCE_INLINE size_t core_total_dims() const {
			return allocated_dims[0] * allocated_dims[1] * allocated_dims[2] * allocated_dims[3];
		}

		RT_TM_FORCE_INLINE size_t core_total_byte_size() const {
			size_t total_elements = core_total_dims();
			size_t block_size	  = core_block_size();
			size_t type_size	  = core_type_size();
			size_t num_blocks	  = (total_elements + block_size - 1) / block_size;
			return num_blocks * type_size;
		}

		RT_TM_INLINE size_t core_block_size() const {
			return get_type_traits(data_type_val).block_size;
		}

		RT_TM_INLINE size_t core_type_size() const {
			return get_type_traits(data_type_val).type_size;
		}

		RT_TM_INLINE size_t core_row_size(int64_t dims_new) const {
			return core_type_size() * dims_new / core_block_size();
		}
	};

	struct core_base {
		array<size_t, 4> allocated_dims{ { 1, 1, 1, 1 } };
		std::vector<core_base*> input_ops{};
		size_t comparison_index{};
		data_type data_type_val{};
		mutable void* data{};
		const char* name{};
		size_t op_id{};
		op_type type{};

		RT_TM_FORCE_INLINE size_t core_total_dims() const {
			return allocated_dims[0] * allocated_dims[1] * allocated_dims[2] * allocated_dims[3];
		}

		RT_TM_FORCE_INLINE size_t core_total_byte_size() const {
			size_t total_elements = core_total_dims();
			size_t block_size	  = core_block_size();
			size_t type_size	  = core_type_size();
			size_t num_blocks	  = (total_elements + block_size - 1) / block_size;
			return num_blocks * type_size;
		}

		RT_TM_INLINE size_t core_block_size() const {
			return get_type_traits(data_type_val).block_size;
		}

		RT_TM_INLINE size_t core_type_size() const {
			return get_type_traits(data_type_val).type_size;
		}

		RT_TM_INLINE size_t core_row_size(int64_t dims_new) const {
			return core_type_size() * dims_new / core_block_size();
		}

	  protected:
		constexpr core_base() noexcept = default;
	};

	template<auto type_new> struct op_core;

	template<auto type_new> struct op_core : public param_api<op_core<type_new>>, public core_base {
		std::vector<core_base*> input_ops{};
		op_core(const core_base_creation_data& other) noexcept {
			this->data_type_val	 = other.data_type_val;
			this->allocated_dims = other.allocated_dims;
			this->op_id			 = other.op_id;
			this->name			 = other.name;
			this->data			 = other.data;
			this->type			 = other.type;
		};
		op_core& operator=(op_core&&) noexcept		= default;
		op_core(op_core&&) noexcept					= default;
		op_core& operator=(const op_core&) noexcept = default;
		op_core(const op_core&) noexcept			= default;
		op_core() noexcept							= default;
		~op_core() noexcept							= default;
	};

	template<> struct op_core<op_type::rope> : public param_api<op_core<op_type::rope>>, public core_base {
		static constexpr op_type type{ op_type::rope };
		uint64_t rope_dimension_count{};
		double rope_freq_base{};
		op_core(const core_base_creation_data& other) noexcept {
			rope_dimension_count = other.get_value<uint64_t, rope_aux_params::rope_dimension_count>();
			rope_freq_base		 = other.get_value<uint64_t, rope_aux_params::rope_freq_base>();
			data_type_val		 = other.data_type_val;
			allocated_dims		 = other.allocated_dims;
			op_id				 = other.op_id;
			name				 = other.name;
			data				 = other.data;
		};
		op_core& operator=(op_core&&) noexcept		= default;
		op_core(op_core&&) noexcept					= default;
		op_core& operator=(const op_core&) noexcept = default;
		op_core(const op_core&) noexcept			= default;
		op_core() noexcept							= default;
		~op_core() noexcept							= default;
	};

	template<> struct op_core<op_type::rms_norm> : public param_api<op_core<op_type::rms_norm>>, public core_base {
		static constexpr op_type type{ op_type::rope };
		std::vector<core_base*> input_ops{};
		uint64_t rms_norm_epsilon{};
		op_core(const core_base_creation_data& other) noexcept {
			rms_norm_epsilon = other.get_value<uint64_t, rms_norm_aux_params::rms_norm_epsilon>();
			data_type_val	 = other.data_type_val;
			allocated_dims	 = other.allocated_dims;
			op_id			 = other.op_id;
			name			 = other.name;
			data			 = other.data;
		};
		op_core& operator=(op_core&&) noexcept		= default;
		op_core(op_core&&) noexcept					= default;
		op_core& operator=(const op_core&) noexcept = default;
		op_core(const op_core&) noexcept			= default;
		op_core() noexcept							= default;
		~op_core() noexcept							= default;
	};

}