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
#include <nihilus/common/param_api.hpp>
#include <nihilus/common/core_traits.hpp>

namespace nihilus {

	struct core_base_creation_data {
		array<uint64_t, 4> allocated_dims{ { 1, 1, 1, 1 } };
		mutable void* data{};
		const char* name{};

		NIHILUS_FORCE_INLINE uint64_t core_total_dims() const {
			return allocated_dims[0] * allocated_dims[1] * allocated_dims[2] * allocated_dims[3];
		}

		NIHILUS_FORCE_INLINE uint64_t core_total_byte_size() const {
			uint64_t total_elements = core_total_dims();
			uint64_t block_size	  = core_block_size();
			uint64_t type_size	  = core_type_size();
			uint64_t num_blocks	  = (total_elements + block_size - 1) / block_size;
			return num_blocks * type_size;
		}

		NIHILUS_INLINE uint64_t core_block_size() const {
			return {};//get_type_traits(data_type_val).block_size;
		}

		NIHILUS_INLINE uint64_t core_type_size() const {
			return {};//get_type_traits(data_type_val).type_size;
		}

		NIHILUS_INLINE uint64_t core_row_size(int64_t dims_new) const {
			return core_type_size() * dims_new / core_block_size();
		}
	};
	/*

	struct core_base {
		mutable array<uint64_t, 4> allocated_dims{ { 1, 1, 1, 1 } };
		std::vector<core_base*> input_ops{};
		uint64_t comparison_index{};
		data_type data_type_val{};
		mutable void* data{};
		const char* name{};
		uint64_t op_id{};
		kernel_type type{};

		NIHILUS_FORCE_INLINE uint64_t core_total_dims() const {
			return allocated_dims[0] * allocated_dims[1] * allocated_dims[2] * allocated_dims[3];
		}

		NIHILUS_FORCE_INLINE uint64_t core_total_byte_size() const {
			uint64_t total_elements = core_total_dims();
			uint64_t block_size	  = core_block_size();
			uint64_t type_size	  = core_type_size();
			uint64_t num_blocks	  = (total_elements + block_size - 1) / block_size;
			return num_blocks * type_size;
		}

		NIHILUS_INLINE uint64_t core_block_size() const {
			return get_type_traits(data_type_val).block_size;
		}

		NIHILUS_INLINE uint64_t core_type_size() const {
			return get_type_traits(data_type_val).type_size;
		}

		NIHILUS_INLINE uint64_t core_row_size(int64_t dims_new) const {
			return core_type_size() * dims_new / core_block_size();
		}

	  protected:
		constexpr core_base() noexcept = default;
	};*/

}