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
#include <rt_tm/cpu/cpu_op_core.hpp>
#include <rt_tm/common/hash.hpp>

namespace rt_tm {

	template<uint64_t type_count> struct type_group {
		array<data_type, type_count> types{};
	};

	template<bool are_we_last_new, size_t index_new> struct type_group_impl {
		inline static constexpr size_t index{ index_new };
		inline static constexpr bool are_we_last{ are_we_last_new };
		RT_TM_FORCE_INLINE bool operator==(const cpu_op_core_thread_base* other) const {
			return index == other->core_base_ptr->comparison_index;
		}
	};

	template<op_type> struct op_entity;

	template<> struct op_entity<op_type::unset> {
		inline static constexpr op_type type{ op_type::unset };
		inline static constexpr size_t type_count{ 1 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::noop> {
		inline static constexpr op_type type{ op_type::noop };
		inline static constexpr size_t type_count{ 1 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::mul_mat> {
		inline static constexpr op_type type{ op_type::mul_mat };
		inline static constexpr size_t type_count{ 3 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::mul> {
		inline static constexpr op_type type{ op_type::mul };
		inline static constexpr size_t type_count{ 3 };
		static constexpr array<type_group<type_count>, 1> groups{ [] {
			array<type_group<type_count>, 1> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::add> {
		inline static constexpr op_type type{ op_type::add };
		inline static constexpr size_t type_count{ 3 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::sub> {
		inline static constexpr op_type type{ op_type::sub };
		inline static constexpr size_t type_count{ 3 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::get_rows> {
		inline static constexpr op_type type{ op_type::get_rows };
		inline static constexpr size_t type_count{ 3 };
		static constexpr array<type_group<type_count>, 1> groups{ [] {
			array<type_group<type_count>, 1> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::view> {
		inline static constexpr op_type type{ op_type::view };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::copy> {
		inline static constexpr op_type type{ op_type::copy };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 2> groups{ [] {
			array<type_group<type_count>, 2> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::softmax> {
		inline static constexpr op_type type{ op_type::softmax };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::rms_norm> {
		inline static constexpr op_type type{ op_type::rms_norm };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::reshape> {
		inline static constexpr op_type type{ op_type::reshape };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::rope> {
		inline static constexpr op_type type{ op_type::rope };
		inline static constexpr size_t type_count{ 4 };
		static constexpr array<type_group<type_count>, 1> groups{ [] {
			array<type_group<type_count>, 1> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::transpose> {
		inline static constexpr op_type type{ op_type::transpose };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::permute> {
		inline static constexpr op_type type{ op_type::permute };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::cont> {
		inline static constexpr op_type type{ op_type::cont };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::silu> {
		inline static constexpr op_type type{ op_type::silu };
		inline static constexpr size_t type_count{ 2 };
		static constexpr array<type_group<type_count>, 0> groups{ [] {
			array<type_group<type_count>, 0> return_values{};
			return return_values;
		}() };
	};

	template<typename op_entity_type> struct get_comparison_value {
		static constexpr auto groups{ op_entity_type::groups };
		template<uint64_t type_count> RT_TM_FORCE_INLINE static const size_t impl(data_type (&type01)[type_count]) {
			for (size_t x = 0; x < groups.size(); ++x) {
				if (std::memcmp(&type01, groups[x].types.data(), groups[x].types.size() * sizeof(data_type)) == 0) {
					return x;
				}
			}
			return std::numeric_limits<size_t>::max();
		}
	};

	template<op_type type_val, size_t entity_index, size_t array_index> struct get_type_from_enum {
		using entity_type			 = op_entity<type_val>;
		static constexpr auto groups = op_entity<type_val>::groups;
		static_assert(entity_index < groups.size(), "Sorry, but that entity index is out of bounds!");
		static_assert(array_index < groups[entity_index].types.size(), "Sorry, but that array index is out of bounds!");
		RT_TM_FORCE_INLINE constexpr static auto get_type() noexcept {
			constexpr auto type = groups[entity_index].types[array_index];
			if constexpr (type == data_type::float_32) {
				return float{};
			}
		}
		using type = decltype(get_type());
	};

	template<device_type dev_type, impl_indices indices_new, op_type type, size_t entity_index> struct kernel_dispatcher {};

	template<size_t entity_index> struct kernel_dispatcher<device_type::cpu, impl_indices{ .cpu_index = 0 }, op_type::rope, entity_index> {
		RT_TM_FORCE_INLINE static void impl(cpu_op_core_thread_base* params_new) {
			auto params					= static_cast<cpu_op_core_thread<3>*>(params_new);
			auto params_rop		= static_cast<op_core<op_type::rope>*>(params->core_base_ptr);
			double rope_freq_base		= params_rop->rope_freq_base;
			using source01_type = get_type_from_enum<op_type::rope, 0, 0>::type;
			using source02_type = get_type_from_enum<op_type::rope, 0, 1>::type;
			using source02_type = get_type_from_enum<op_type::rope, 0, 2>::type;
			using dst_type		= get_type_from_enum<op_type::rope, 0, 3>::type;
		}
	};

	template<typename... bases> struct op_map : public bases... {
		template<typename op_entity_type> RT_TM_FORCE_INLINE static bool iterate_values_impl(cpu_op_core_thread_base* params) {
			return op_entity_type::processIndex(params);
		}

		static constexpr void iterate_values(cpu_op_core_thread_base* params) {
			(iterate_values_impl<bases>(params) && ...);
		}
	};

	template<device_type dev_type, impl_indices indices_new, op_type type, typename data_type_entity_type> struct data_type_types {
		constexpr data_type_types() noexcept = default;

		RT_TM_FORCE_INLINE static bool processIndex(cpu_op_core_thread_base* params) {
			if (params->core_base_ptr->comparison_index == data_type_entity_type::index) {
				//kernel_dispatcher<dev_type, indices_new, type, data_type_entity_type::index>;
				return false;
			} else {
				if constexpr (data_type_entity_type::are_we_last) {
					std::string error_string{ "Sorry, but you need to create a 'type_group' value to be stored in the array of op_entity, for op_type: " +
						static_cast<std::string>(print_enum_value<static_cast<op_type>(0), op_type::count>(type)) };
					if (params->core_base_ptr->input_ops.size() > 0) {
						error_string += ", for data_types: ";
					}
					for (size_t x = 0; x < params->core_base_ptr->input_ops.size(); ++x) {
						error_string += print_enum_value<static_cast<data_type>(0), data_type::count>(params->core_base_ptr->input_ops[x]->data_type_val);
						error_string += ", ";
					}
					error_string += "and (dst-type) == ";
					error_string += print_enum_value<static_cast<data_type>(0), data_type::count>(params->core_base_ptr->data_type_val);
					std::cerr << error_string << std::endl;
				}
				return true;
			}
		};
	};

	template<device_type dev_type, impl_indices indices_new, op_type type, typename index_sequence> struct get_data_type_type_base;

	template<device_type dev_type, impl_indices indices_new, op_type type_val, uint64_t... index>
	struct get_data_type_type_base<dev_type, indices_new, type_val, std::index_sequence<index...>> {
		using type = op_map<data_type_types<dev_type, indices_new, type_val, type_group_impl<index == op_entity<static_cast<op_type>(type_val)>::groups.size() - 1, index>>...>;
	};

	template<device_type dev_type, impl_indices indices_new, op_type type> using data_type_type_base_t =
		typename get_data_type_type_base<dev_type, indices_new, type, std::make_index_sequence<op_entity<type>::groups.size()>>::type;

	template<device_type dev_type, op_type type, impl_indices indices_new> struct data_type_dispatcher_final {
		RT_TM_FORCE_INLINE static void impl(cpu_op_core_thread_base* params) {
			data_type_type_base_t<dev_type, indices_new, type>::iterate_values(params);
		}
	};

	template<typename op_entity_type, device_type dev_type, impl_indices indices_new> struct op_types {
		constexpr op_types() noexcept = default;

		RT_TM_FORCE_INLINE static bool processIndex(cpu_op_core_thread_base* params) {
			if (params->core_base_ptr->type == op_entity_type::type) {
				data_type_dispatcher_final<dev_type, op_entity_type::type, indices_new>::impl(params);
				return false;
			} else {
				return true;
			}
		};
	};

	template<template<auto...> typename op_entity_type, template<typename, auto...> typename op_types, typename index_sequence, auto... types> struct get_entity_type_base;

	template<template<auto...> typename op_entity_type, template<typename, auto...> typename op_types, uint64_t... index, auto... types>
	struct get_entity_type_base<op_entity_type, op_types, std::index_sequence<index...>, types...> {
		using type = op_map<op_types<op_entity_type<static_cast<op_type>(index)>, types...>...>;
	};

	template<device_type dev_type, impl_indices indices_new> using op_type_base_t =
		typename get_entity_type_base<op_entity, op_types, std::make_index_sequence<static_cast<uint64_t>(op_type::count)>, dev_type, indices_new>::type;

	template<device_type dev_type, impl_indices indices_new> struct op_dispatcher_final {
		RT_TM_FORCE_INLINE static void impl(cpu_op_core_thread_base* params) {
			op_type_base_t<dev_type, indices_new>::iterate_values(params);
		}
	};
}