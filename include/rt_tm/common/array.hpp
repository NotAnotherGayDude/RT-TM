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

#include <rt_tm/common/iterator.hpp>
#include <algorithm>
#include <stdexcept>

namespace rt_tm {

	template<typename value_type_new, size_t size_new> struct array {
	  public:
		static constexpr size_t size_val{ size_new };
		using value_type			 = value_type_new;
		using size_type				 = size_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = array_iterator<value_type, size_new>;
		using const_iterator		 = const array_iterator<value_type, size_new>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		RT_TM_FORCE_INLINE constexpr array(){};

		RT_TM_FORCE_INLINE constexpr array(std::initializer_list<value_type> list) {
			for (size_t x = 0; x < list.size(); ++x) {
				data_val[x] = list.begin()[x];
			}
		}

		RT_TM_FORCE_INLINE constexpr void fill(const value_type& _Value) {
			std::fill_n(data_val, size_new, _Value);
		}

		RT_TM_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator(data_val);
		}

		RT_TM_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		RT_TM_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator(data_val + size_val);
		}

		RT_TM_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator(data_val + size_val);
		}

		RT_TM_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		RT_TM_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		RT_TM_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		RT_TM_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		RT_TM_FORCE_INLINE constexpr size_type size() const noexcept {
			return size_new;
		}

		RT_TM_FORCE_INLINE constexpr size_type max_size() const noexcept {
			return size_new;
		}

		RT_TM_FORCE_INLINE constexpr bool empty() const noexcept {
			return false;
		}

		RT_TM_FORCE_INLINE constexpr reference at(size_type position) {
			if (size_new <= position) {
				throw std::runtime_error{ "invalid array<T, N> subscript" };
			}

			return data_val[position];
		}

		RT_TM_FORCE_INLINE constexpr const_reference at(size_type position) const {
			if (size_new <= position) {
				throw std::runtime_error{ "invalid array<T, N> subscript" };
			}

			return data_val[position];
		}

		RT_TM_FORCE_INLINE constexpr reference operator[](size_type position) noexcept {
			return data_val[position];
		}

		RT_TM_FORCE_INLINE constexpr const_reference operator[](size_type position) const noexcept {
			return data_val[position];
		}

		RT_TM_FORCE_INLINE constexpr reference front() noexcept {
			return data_val[0];
		}

		RT_TM_FORCE_INLINE constexpr const_reference front() const noexcept {
			return data_val[0];
		}

		RT_TM_FORCE_INLINE constexpr reference back() noexcept {
			return data_val[size_new - 1];
		}

		RT_TM_FORCE_INLINE constexpr const_reference back() const noexcept {
			return data_val[size_new - 1];
		}

		RT_TM_FORCE_INLINE constexpr value_type* data() noexcept {
			return data_val;
		}

		RT_TM_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return data_val;
		}

		RT_TM_FORCE_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
			for (size_t x = 0; x < size_val; ++x) {
				if (lhs[x] != rhs[x]) {
					return false;
				}
			}
			return true;
		}

		value_type data_val[size_val]{};
	};

	template<typename T, typename... U> array(T, U...) -> array<T, 1 + sizeof...(U)>;

	struct empty_array_element {};

	template<class value_type_new> class array<value_type_new, 0> {
	  public:
		using value_type			 = value_type_new;
		using size_type				 = size_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = array_iterator<value_type, 0>;
		using const_iterator		 = const array_iterator<value_type, 0>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		RT_TM_FORCE_INLINE constexpr void fill(const value_type&) {
		}

		RT_TM_FORCE_INLINE constexpr void swap(array&) noexcept {
		}

		RT_TM_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator{};
		}

		RT_TM_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator{};
		}

		RT_TM_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator{};
		}

		RT_TM_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator{};
		}

		RT_TM_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		RT_TM_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		RT_TM_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		RT_TM_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		RT_TM_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		RT_TM_FORCE_INLINE constexpr size_type size() const noexcept {
			return 0;
		}

		RT_TM_FORCE_INLINE constexpr size_type max_size() const noexcept {
			return 0;
		}

		RT_TM_FORCE_INLINE constexpr bool empty() const noexcept {
			return true;
		}

		RT_TM_FORCE_INLINE constexpr reference at(size_type) {
			throw std::runtime_error{ "invalid array<T, N> subscript" };
		}

		RT_TM_FORCE_INLINE constexpr const_reference at(size_type) const {
			throw std::runtime_error{ "invalid array<T, N> subscript" };
		}

		RT_TM_FORCE_INLINE constexpr reference operator[](size_type) noexcept {
			return *data();
		}

		RT_TM_FORCE_INLINE constexpr const_reference operator[](size_type) const noexcept {
			return *data();
		}

		RT_TM_FORCE_INLINE constexpr reference front() noexcept {
			return *data();
		}

		RT_TM_FORCE_INLINE constexpr const_reference front() const noexcept {
			return *data();
		}

		RT_TM_FORCE_INLINE constexpr reference back() noexcept {
			return *data();
		}

		RT_TM_FORCE_INLINE constexpr const_reference back() const noexcept {
			return *data();
		}

		RT_TM_FORCE_INLINE constexpr value_type* data() noexcept {
			return nullptr;
		}

		RT_TM_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return nullptr;
		}

		RT_TM_FORCE_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
			( void )lhs;
			( void )rhs;
			return true;
		}

	  private:
		std::conditional_t<std::disjunction_v<std::is_default_constructible<value_type>, std::is_default_constructible<value_type>>, value_type, empty_array_element> data_val[1]{};
	};
}
