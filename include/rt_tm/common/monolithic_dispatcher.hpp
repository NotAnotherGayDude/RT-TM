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

	template<impl_indices indices, bool are_we_last_new, size_t index_new, data_type... types_new> struct type_group_impl {
		inline static constexpr array<data_type, sizeof...(types_new)> types{ types_new... };
		inline static constexpr size_t index{ index_new };
		inline static constexpr bool are_we_last{ are_we_last_new };
		RT_TM_FORCE_INLINE bool operator==(const cpu_op_core_thread_base* other) const {
			return index == other->core_base_ptr->comparison_index;
		}
	};

	template<op_type> struct op_entity;

	template<> struct op_entity<op_type::unset> {
		inline static constexpr op_type type{ op_type::unset };
		static constexpr array<type_group<1>, 0> groups{ [] {
			array<type_group<1>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::noop> {
		inline static constexpr op_type type{ op_type::noop };
		static constexpr array<type_group<1>, 0> groups{ [] {
			array<type_group<1>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::mul_mat> {
		inline static constexpr op_type type{ op_type::mul_mat };
		static constexpr array<type_group<3>, 0> groups{ [] {
			array<type_group<3>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::mul> {
		inline static constexpr op_type type{ op_type::mul };
		static constexpr array<type_group<3>, 0> groups{ [] {
			array<type_group<3>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::add> {
		inline static constexpr op_type type{ op_type::add };
		static constexpr array<type_group<3>, 0> groups{ [] {
			array<type_group<3>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::sub> {
		inline static constexpr op_type type{ op_type::sub };
		static constexpr array<type_group<3>, 0> groups{ [] {
			array<type_group<3>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::get_rows> {
		inline static constexpr op_type type{ op_type::get_rows };
		static constexpr array<type_group<3>, 0> groups{ [] {
			array<type_group<3>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::view> {
		inline static constexpr op_type type{ op_type::view };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::copy> {
		inline static constexpr op_type type{ op_type::copy };
		static constexpr array<type_group<2>, 2> groups{ [] {
			array<type_group<2>, 2> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::softmax> {
		inline static constexpr op_type type{ op_type::softmax };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::rms_norm> {
		inline static constexpr op_type type{ op_type::rms_norm };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::reshape> {
		inline static constexpr op_type type{ op_type::reshape };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::rope> {
		inline static constexpr op_type type{ op_type::rope };
		static constexpr array<type_group<4>, 0> groups{ [] {
			array<type_group<4>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::transpose> {
		inline static constexpr op_type type{ op_type::transpose };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::permute> {
		inline static constexpr op_type type{ op_type::permute };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::cont> {
		inline static constexpr op_type type{ op_type::cont };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
			return return_values;
		}() };
	};

	template<> struct op_entity<op_type::silu> {
		inline static constexpr op_type type{ op_type::silu };
		static constexpr array<type_group<2>, 0> groups{ [] {
			array<type_group<2>, 0> return_values{};
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
			return 0;
		}
	};

	template<template<impl_indices, op_type, data_type, data_type, data_type> typename function_type, device_type dev_type, impl_indices indices_new, op_type op_type_val>
	struct op_dispatcher;

	template<template<impl_indices, op_type, data_type, data_type, data_type> typename function_type, device_type dev_type, impl_indices indices_new, op_type op_type_val,
		typename op_entity_type>
	struct op_entities_internal : public op_entity_type {
		constexpr op_entities_internal() noexcept = default;

		RT_TM_FORCE_INLINE static bool processIndex(cpu_op_core_thread_base* params) {
			static constexpr op_entity_type op_entity{};
			if (params == op_entity) {
				//function_type<indices_new, op_type, op_entity_type::type01, op_entity_type::type02, op_entity_type::type03>::impl(params);
				return false;
			} else {
				if constexpr (op_entity_type::are_we_last) {
					std::string error_string{ "Sorry, but you need to create a 'type_trio' value to be stored in the array of op_entity, for op_type: " +
						static_cast<std::string>(print_enum_value<static_cast<op_type>(0), op_type::count>(op_type_val)) + ", for data_types: " };
					for (uint64_t x = 0; x < params->core_base_ptr->input_ops.size(); ++x) {
						error_string += print_enum_value<static_cast<data_type>(0), data_type::count>(params->core_base_ptr->input_ops[x]->data_type_val);
						error_string += ", ";
					}
					error_string += "and (dst-type) == ";
					error_string += print_enum_value<static_cast<data_type>(0), data_type::count>(params->core_base_ptr->data_type_val);
				}
				return true;
			}
		};
	};

	template<impl_indices indices_new, typename... bases> struct op_map : public bases... {
		template<typename op_entity_type> RT_TM_FORCE_INLINE static bool iterate_values_impl(cpu_op_core_thread_base* params) {
			return op_entity_type::processIndex(params);
		}

		static constexpr void iterate_values(cpu_op_core_thread_base* params) {
			(iterate_values_impl<bases>(params) && ...);
		}
	};

	template<impl_indices indices_new, op_type type, uint64_t, typename index_sequence> struct op_entity_getter;

	template<impl_indices indices_new, op_type op_type_val, uint64_t index, uint64_t... indices>
	struct op_entity_getter<indices_new, op_type_val, index, std::index_sequence<indices...>> {
		static constexpr auto index_count = op_entity<op_type_val>::groups.size();
		static constexpr auto group		  = op_entity<op_type_val>::groups[index];
		using type						  = type_group_impl<indices_new, index, index == (index_count - 1), group.types[indices]...>;
	};

	template<template<impl_indices, op_type, data_type, data_type, data_type> typename function_type, device_type dev_type, impl_indices indices_new, op_type op_type_val,
		typename index_sequence, typename... value_types>
	struct get_op_entity_base_internal;

	template<template<impl_indices, op_type, data_type, data_type, data_type> typename function_type, device_type dev_type, impl_indices indices_new, op_type op_type_val,
		uint64_t... index>
	struct get_op_entity_base_internal<function_type, dev_type, indices_new, op_type_val, std::index_sequence<index...>> {
		using type = op_map<indices_new,
			op_entities_internal<function_type, dev_type, indices_new, op_type_val,
				typename op_entity_getter<indices_new, op_type_val, index, std::make_index_sequence<sizeof...(index)>>::type>...>;
	};

	template<template<impl_indices, op_type, data_type, data_type, data_type> typename function_type, device_type dev_type, impl_indices indices_new, op_type op_type_val>
	using op_entity_base_internal_t = typename get_op_entity_base_internal<function_type, dev_type, indices_new, op_type_val,
		std::make_index_sequence<static_cast<uint64_t>(op_entity<op_type_val>::groups.size())>>::type;

	template<template<impl_indices, op_type, data_type, data_type, data_type> typename function_type, device_type dev_type, impl_indices indices_new, op_type op_type_val>
	struct op_dispatcher {
		template<typename... arg_types> RT_TM_FORCE_INLINE static void impl(arg_types&&... params) {
			std::cout << "CURRENT TYPE: " << ( int32_t )op_type_val << std::endl;
			op_entity_base_internal_t<function_type, dev_type, indices_new, op_type_val>::iterate_values(std::forward<arg_types>(params)...);
		}
	};
	template<typename value_type>
	concept specializable = requires(value_type value) { value_type::specialized; };

	template<impl_indices indices, op_type op_type, data_type... types> struct function_dispatcher_new {
		RT_TM_FORCE_INLINE static void impl(cpu_op_core_thread_base* params) {
			std::cout << "CURRENT TYPE: " << ( int32_t )op_type << std::endl;
			//using function_dispatcher_type = function_dispatcher_impl<op_type, indices.cpu_index, types...>;
			//if constexpr (specializable<function_dispatcher_type>) {
				//return function_dispatcher_impl<op_type, indices.cpu_index, type...>::impl(params);
			//} else {//
			//static_assert(false, "Sorry, but you need to add a \"specialized\" boolean to the type.");
			//}
		}
	};

	template<device_type dev_type, impl_indices indices_new, typename op_entity_type> struct op_types : public op_entity_type {
		constexpr op_types() noexcept = default;

		RT_TM_FORCE_INLINE static bool processIndex(cpu_op_core_thread_base* params) {
			if (params->core_base_ptr->type == op_entity_type::type) {
				std::cout << "CURRENT TYPE: " << ( int32_t )op_entity_type::type << std::endl;
				op_dispatcher<function_dispatcher_new, dev_type, indices_new, op_entity_type::type>::impl(params);
				return false;
			} else {
				return true;
			}
		};
	};

	template<device_type dev_type, impl_indices indices_new, typename index_sequence, typename... value_types> struct get_op_type_base;

	template<device_type dev_type, impl_indices indices_new, uint64_t... index> struct get_op_type_base<dev_type, indices_new, std::index_sequence<index...>> {
		using type = op_map<indices_new, op_types<dev_type, indices_new, op_entity<static_cast<op_type>(index)>>...>;
	};

	template<device_type dev_type, impl_indices indices_new> using op_type_base_t =
		typename get_op_type_base<dev_type, indices_new, std::make_index_sequence<static_cast<uint64_t>(op_type::count)>>::type;

	template<device_type dev_type, impl_indices indices_new> struct op_dispatcher_final {
		RT_TM_FORCE_INLINE static void impl(cpu_op_core_thread_base* params) {
			op_type_base_t<dev_type, indices_new>::iterate_values(params);
		}
	};

}
