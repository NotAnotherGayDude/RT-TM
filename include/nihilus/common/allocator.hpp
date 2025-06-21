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

#include <nihilus/common/config.hpp>
#include <memory_resource>

namespace nihilus {

	template<auto multiple, typename value_type01 = decltype(multiple)> NIHILUS_FORCE_INLINE constexpr value_type01 round_up_to_multiple(value_type01 value) noexcept {
		if constexpr ((multiple & (multiple - 1)) == 0) {
			constexpr auto mulSub1{ multiple - 1 };
			auto notMulSub1{ ~mulSub1 };
			return (value + (mulSub1)) & notMulSub1;
		} else {
			const auto remainder = value % multiple;
			return remainder == 0 ? value : value + (multiple - remainder);
		}
	}

	template<typename value_type_new> class allocator {
	  public:
		using value_type	   = value_type_new;
		using pointer		   = value_type_new*;
		using const_pointer	   = const value_type_new*;
		using reference		   = value_type_new&;
		using const_reference  = const value_type_new&;
		using size_type		   = std::uint64_t;
		using difference_type  = std::ptrdiff_t;
		using allocator_traits = std::allocator_traits<allocator<value_type>>;

		template<typename U> struct rebind {
			using other = allocator<U>;
		};

		NIHILUS_FORCE_INLINE allocator() noexcept = default;

		template<typename U> allocator(const allocator<U>&) noexcept {
		}

		NIHILUS_FORCE_INLINE static pointer allocate(size_type count_new) noexcept {
			if NIHILUS_UNLIKELY (count_new == 0) {
				return nullptr;
			}
#if defined(NIHILUS_PLATFORM_WINDOWS) || defined(NIHILUS_PLATFORM_LINUX)
			return static_cast<value_type*>(_mm_malloc(round_up_to_multiple<cpu_alignment>(count_new * sizeof(value_type)), cpu_alignment));
#else
			return static_cast<value_type*>(aligned_alloc(cpu_alignment, round_up_to_multiple<cpu_alignment>(count_new * sizeof(value_type))));
#endif
		}

		NIHILUS_FORCE_INLINE static void deallocate(pointer ptr, uint64_t = 0) noexcept {
			if NIHILUS_LIKELY (ptr) {
#if defined(NIHILUS_PLATFORM_WINDOWS) || defined(NIHILUS_PLATFORM_LINUX)
				_mm_free(ptr);
#else
				free(ptr);
#endif
			}
		}

		template<typename... arg_types> NIHILUS_FORCE_INLINE static void construct(pointer ptr, arg_types&&... args) noexcept {
			new (ptr) value_type(std::forward<arg_types>(args)...);
		}

		NIHILUS_FORCE_INLINE static size_type maxSize() noexcept {
			return allocator_traits::max_size(allocator{});
		}

		NIHILUS_FORCE_INLINE static void destroy(pointer ptr) noexcept {
			ptr->~value_type();
		}
	};

}// namespace internal