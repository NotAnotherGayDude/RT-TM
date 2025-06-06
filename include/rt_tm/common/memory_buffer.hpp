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

#include <rt_tm/common/allocator.hpp>
#include <rt_tm/common/common.hpp>
#include <rt_tm/common/config.hpp>
#include <stdexcept>
#include <iterator>

namespace rt_tm {

	template<global_config config> struct memory_buffer : public allocator<uint8_t> {
		using value_type = uint8_t;
		using alloc		 = allocator<value_type>;
		using pointer	 = value_type*;
		using size_type	 = size_t;

		RT_TM_FORCE_INLINE memory_buffer() noexcept = default;

		RT_TM_FORCE_INLINE memory_buffer& operator=(const memory_buffer&) noexcept = delete;
		RT_TM_FORCE_INLINE memory_buffer(const memory_buffer&) noexcept			   = delete;

		RT_TM_FORCE_INLINE memory_buffer& operator=(memory_buffer&& other) noexcept {
			if (this != &other) {
				std::swap(current_offset, other.current_offset);
				std::swap(data_val, other.data_val);
				std::swap(size_val, other.size_val);
			}
			return *this;
		}

		RT_TM_FORCE_INLINE memory_buffer(memory_buffer&& other) noexcept {
			*this = std::move(other);
		}

		RT_TM_FORCE_INLINE void init(size_t size) noexcept {
			if (data_val) {
				clear();
			}
			data_val = alloc::allocate(size);
			size_val = size;
		}

		RT_TM_FORCE_INLINE void clear() noexcept {
			if (data_val) {
				alloc::deallocate(data_val);
				data_val = nullptr;
				size_val = 0;
			}
		}

		RT_TM_FORCE_INLINE pointer data() noexcept {
			return data_val;
		}

		RT_TM_FORCE_INLINE pointer claim_memory(size_t amount_to_claim) noexcept {
			if (current_offset + amount_to_claim > size_val) {
				if constexpr (config.exceptions) {
					throw std::runtime_error{ "Sorry, but this memory_buffer is out of memory!" };
				} else {
					return nullptr;
				}
			}
			pointer return_value = data_val + current_offset;
			current_offset += amount_to_claim;
			return return_value;
		}

		RT_TM_FORCE_INLINE ~memory_buffer() noexcept {
			if (data_val && size_val > 0) {
				alloc::deallocate(data_val);
				data_val = nullptr;
			}
		}

	  protected:
		size_type current_offset{};
		value_type* data_val{};
		size_type size_val{};
	};

}
