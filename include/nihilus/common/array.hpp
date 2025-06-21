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

#include <nihilus/common/iterator.hpp>
#include <nihilus/common/config.hpp>
#include <algorithm>
#include <stdexcept>

namespace nihilus {

	template<typename value_type01, typename value_type02> struct is_indexable {
		static constexpr bool indexable{ std::is_same_v<value_type01, value_type02> || std::integral<value_type01> };
	};

	enum class array_static_assert_errors {
		invalid_index_type,
	};

	template<typename value_type_new, auto size_new> struct array {
	  public:
		static_assert(integral_or_enum<decltype(size_new)>, "Sorry, but the size val passed to array must be integral or enum!");
		static constexpr uint64_t size_val{ static_cast<uint64_t>(size_new) };
		using value_type			 = value_type_new;
		using size_type				 = decltype(size_new);
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = array_iterator<value_type, static_cast<uint64_t>(size_new)>;
		using const_iterator		 = const array_iterator<value_type, static_cast<uint64_t>(size_new)>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		NIHILUS_FORCE_INLINE constexpr array& operator=(const array& other)
			requires(std::copyable<value_type>)
		{
			std::copy(other.data(), other.data() + other.size(), data());
			return *this;
		};

		NIHILUS_FORCE_INLINE constexpr array(const array& other)
			requires(std::copyable<value_type>)
		{
			*this = other;
		};

		NIHILUS_FORCE_INLINE constexpr array(){};

		NIHILUS_FORCE_INLINE constexpr array(std::initializer_list<value_type> list) {
			for (uint64_t x = 0; x < list.size(); ++x) {
				data_val[x] = list.begin()[x];
			}
		}

		NIHILUS_FORCE_INLINE constexpr void fill(const value_type& _Value) {
			std::fill_n(data_val, size_new, _Value);
		}

		NIHILUS_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator(data_val);
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		NIHILUS_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator(data_val + size_val);
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator(data_val + size_val);
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		NIHILUS_FORCE_INLINE constexpr size_type size() const noexcept {
			return size_new;
		}

		NIHILUS_FORCE_INLINE constexpr size_type max_size() const noexcept {
			return size_new;
		}

		NIHILUS_FORCE_INLINE constexpr bool empty() const noexcept {
			return false;
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr reference at(index_type position) {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>::indexable, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			if (size_new <= position) {
				throw std::runtime_error{ "invalid array<T, N> subscript" };
			}

			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr const_reference at(index_type position) const {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>::indexable, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			if (size_new <= position) {
				throw std::runtime_error{ "invalid array<T, N> subscript" };
			}

			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr reference operator[](index_type position) noexcept {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>::indexable, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr const_reference operator[](index_type position) const noexcept {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>::indexable, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			return data_val[static_cast<uint64_t>(position)];
		}

		NIHILUS_FORCE_INLINE constexpr reference front() noexcept {
			return data_val[0];
		}

		NIHILUS_FORCE_INLINE constexpr const_reference front() const noexcept {
			return data_val[0];
		}

		NIHILUS_FORCE_INLINE constexpr reference back() noexcept {
			return data_val[size_new - 1];
		}

		NIHILUS_FORCE_INLINE constexpr const_reference back() const noexcept {
			return data_val[size_new - 1];
		}

		NIHILUS_FORCE_INLINE constexpr value_type* data() noexcept {
			return data_val;
		}

		NIHILUS_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return data_val;
		}

		NIHILUS_FORCE_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
			for (uint64_t x = 0; x < size_val; ++x) {
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
		using size_type				 = uint64_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = array_iterator<value_type, 0>;
		using const_iterator		 = const array_iterator<value_type, 0>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		NIHILUS_FORCE_INLINE constexpr void fill(const value_type&) {
		}

		NIHILUS_FORCE_INLINE constexpr void swap(array&) noexcept {
		}

		NIHILUS_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		NIHILUS_FORCE_INLINE constexpr size_type size() const noexcept {
			return 0;
		}

		NIHILUS_FORCE_INLINE constexpr size_type max_size() const noexcept {
			return 0;
		}

		NIHILUS_FORCE_INLINE constexpr bool empty() const noexcept {
			return true;
		}

		NIHILUS_FORCE_INLINE constexpr reference at(size_type) {
			throw std::runtime_error{ "invalid array<T, N> subscript" };
		}

		NIHILUS_FORCE_INLINE constexpr const_reference at(size_type) const {
			throw std::runtime_error{ "invalid array<T, N> subscript" };
		}

		NIHILUS_FORCE_INLINE constexpr reference operator[](size_type) noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr const_reference operator[](size_type) const noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr reference front() noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr const_reference front() const noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr reference back() noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr const_reference back() const noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr value_type* data() noexcept {
			return nullptr;
		}

		NIHILUS_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return nullptr;
		}

		NIHILUS_FORCE_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
			( void )lhs;
			( void )rhs;
			return true;
		}

	  private:
		std::conditional_t<std::disjunction_v<std::is_default_constructible<value_type>, std::is_default_constructible<value_type>>, value_type, empty_array_element> data_val[1]{};
	};
}
