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
#include <rt_tm/common/input_session.hpp>
#include <rt_tm/common/memory_buffer.hpp>
#include <rt_tm/common/arch_traits.hpp>
#include <rt_tm/cpu/device.hpp>

namespace rt_tm {

	enum class model_name { Meta_Llama_3_1_8B_Instruct_Q8_0_gguf };

	template<model_arch arch, auto depth_new> struct op_graph_bases_op;

	template<auto type_new> struct op_graph_bases_op<model_arch::llama, type_new> {
		using enum_type																	   = decltype(type_new);
		RT_TM_FORCE_INLINE op_graph_bases_op() noexcept									   = default;
		RT_TM_FORCE_INLINE op_graph_bases_op& operator=(const op_graph_bases_op&) noexcept = delete;
		RT_TM_FORCE_INLINE op_graph_bases_op(const op_graph_bases_op&) noexcept			   = delete;
		//std::vector<core<static_cast<llama_op_names>(type_new)>> ops{};
		RT_TM_FORCE_INLINE core_base* emplace_back(enum_type args) {
			if (args == type_new) {
				//auto& new_ref = ops.emplace_back();
				return nullptr;
				//&new_ref;
			} else {
				return nullptr;
			}
		}
	};

	template<typename... bases> struct op_graph_bases : public bases... {
		RT_TM_FORCE_INLINE op_graph_bases() noexcept						= default;
		RT_TM_FORCE_INLINE op_graph_bases& operator=(op_graph_bases&&)		= delete;
		RT_TM_FORCE_INLINE op_graph_bases(op_graph_bases&&)					= delete;
		RT_TM_FORCE_INLINE op_graph_bases& operator=(const op_graph_bases&) = delete;
		RT_TM_FORCE_INLINE op_graph_bases(const op_graph_bases&)			= delete;

		template<typename op_entity_type, typename enum_type> RT_TM_FORCE_INLINE core_base* emplace_back(enum_type args) {
			return op_entity_type::emplace_back(args);
		}

		template<typename enum_type> RT_TM_FORCE_INLINE core_base* emplace_back(enum_type args) {
			core_base* result = nullptr;

			((result = emplace_back_impl<bases>(args)) || ...);

			return result;
		}

	  private:
		template<typename base_type, typename enum_type> RT_TM_FORCE_INLINE core_base* emplace_back_impl(enum_type args) {
			if constexpr (requires { base_type::emplace_back(args); }) {
				return base_type::emplace_back(args);
			} else {
				return nullptr;
			}
		}
	};

	template<model_config config, typename index_sequence> struct get_op_graph_bases;

	template<model_config config, size_t... index> struct get_op_graph_bases<config, std::index_sequence<index...>> {
		using type = op_graph_bases<op_graph_bases_op<config.arch, index>...>;
	};

	template<model_config config, typename enum_tyoe> using op_graph_bases_t =
		typename get_op_graph_bases<config, std::make_index_sequence<static_cast<size_t>(enum_tyoe::count)>>::type;

	template<model_config config, typename arch_traits, impl_indices indices_new> struct op_graph_base;

	template<model_config config, impl_indices indices_new> struct op_graph_base<config, arch_traits<model_arch::llama>, indices_new>
		: public device_registry<device_type::cpu, arch_traits<model_arch::llama>, config, indices_new>,
		  op_graph_bases_t<config, arch_traits<model_arch::llama>::enum_type> {
	  public:
		inline static constexpr impl_indices indices{ indices_new };
		op_graph_config config_val{};

		RT_TM_INLINE op_graph_base() noexcept								 = default;
		RT_TM_INLINE op_graph_base& operator=(const op_graph_base&) noexcept = delete;
		RT_TM_INLINE op_graph_base(const op_graph_base&) noexcept			 = delete;

		RT_TM_FORCE_INLINE op_graph_base(op_graph_config graph_config)
			: device_registry<device_type::cpu, arch_traits<model_arch::llama>, config, indices_new>{ graph_config.num_threads }, config_val{ graph_config } {};

		RT_TM_FORCE_INLINE void process_input(input_session& session) {
			reset_state();
			execute_tasks();
		};

		RT_TM_FORCE_INLINE void reset_state() {
			auto& cpu_devices{ static_cast<device_registry<device_type::cpu, arch_traits<model_arch::llama>, config, indices_new>*>(this)->get_devices() };
			for (auto& value: cpu_devices) {
				value->reset_state();
			}
		}

		RT_TM_FORCE_INLINE void schedule_execution() {
			auto& cpu_devices{ static_cast<device_registry<device_type::cpu, arch_traits<model_arch::llama>, config, indices_new>*>(this)->get_devices() };
			for (auto& value: cpu_devices) {
				value->schedule_execution(ops);
			}
		}

		RT_TM_FORCE_INLINE void execute_tasks() {
			auto& cpu_devices{ static_cast<device_registry<device_type::cpu, arch_traits<model_arch::llama>, config, indices_new>*>(this)->get_devices() };
			for (auto& value: cpu_devices) {
				value->execute_tasks();
			}
		}

		RT_TM_FORCE_INLINE ~op_graph_base() {
		}

	  protected:
		std::vector<core_base*> ops{};
	};

	template<model_config config> struct op_graph {
		RT_TM_FORCE_INLINE op_graph& operator=(const op_graph&) = delete;
		RT_TM_FORCE_INLINE op_graph(op_graph&)					= delete;
		RT_TM_FORCE_INLINE op_graph& operator=(op_graph&& other) {
			this->op_graph00.swap(other.op_graph00);
			this->op_graph01.swap(other.op_graph01);
			this->op_graph02.swap(other.op_graph02);
			return *this;
		}

		RT_TM_FORCE_INLINE op_graph(op_graph&& other) {
			*this = std::move(other);
		}

		op_graph(op_graph_config graph_config) {
			switch (cpu_arch_index_holder::cpu_arch_index) {
				case 0ull: {
					op_graph00 = std::make_unique<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 0ull }>>(graph_config);
					break;
				}
				case 1ull: {
					op_graph01 = std::make_unique<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 1ull }>>(graph_config);
					break;
				}
				case 2ull: {
					op_graph02 = std::make_unique<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 2ull }>>(graph_config);
					break;
				}
			}
		}

		RT_TM_FORCE_INLINE void schedule_execution() {
			switch (cpu_arch_index_holder::cpu_arch_index) {
				case 0ull: {
					op_graph00->schedule_execution();
					break;
				}
				case 1ull: {
					op_graph01->schedule_execution();
					break;
				}
				case 2ull: {
					op_graph02->schedule_execution();
					break;
				}
			}
		}

		RT_TM_FORCE_INLINE void reset_state() {
			switch (cpu_arch_index_holder::cpu_arch_index) {
				case 0ull: {
					op_graph00->reset_state();
					break;
				}
				case 1ull: {
					op_graph01->reset_state();
					break;
				}
				case 2ull: {
					op_graph02->reset_state();
					break;
				}
			}
		}

		RT_TM_FORCE_INLINE void process_input(input_session& session) {
			switch (cpu_arch_index_holder::cpu_arch_index) {
				case 0ull: {
					op_graph00->process_input(session);
					break;
				}
				case 1ull: {
					op_graph01->process_input(session);
					break;
				}
				case 2ull: {
					op_graph02->process_input(session);
					break;
				}
			}
		}

		RT_TM_FORCE_INLINE void execute_tasks() {
			switch (cpu_arch_index_holder::cpu_arch_index) {
				case 0ull: {
					op_graph00->execute_tasks();
					break;
				}
				case 1ull: {
					op_graph01->execute_tasks();
					break;
				}
				case 2ull: {
					op_graph02->execute_tasks();
					break;
				}
			}
		}

		RT_TM_FORCE_INLINE void init(op_graph_config graph_config = {}) {
			switch (cpu_arch_index_holder::cpu_arch_index) {
				case 0ull: {
					op_graph00 = std::make_unique<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 0ull }>>(graph_config);
					break;
				}
				case 1ull: {
					op_graph01 = std::make_unique<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 1ull }>>(graph_config);
					break;
				}
				case 2ull: {
					op_graph02 = std::make_unique<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 2ull }>>(graph_config);
					break;
				}
			}
		}

	  protected:
		std::unique_ptr<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 0ull }>> op_graph00{};
		std::unique_ptr<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 1ull }>> op_graph01{};
		std::unique_ptr<op_graph_base<config, arch_traits<config.arch>, impl_indices{ .cpu_index = 2ull }>> op_graph02{};
	};

}