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
#include <rt_tm/cpu/thread_pool.hpp>

namespace rt_tm {

	template<device_type dev, typename derived_type, model_config config, impl_indices indices> struct device;

	template<device_type dev, typename derived_type, model_config config, impl_indices indices> struct device_registry;

	template<model_config config, typename derived_type, impl_indices indices> struct device<device_type::cpu, derived_type, config, indices> {
		using thread_pool_t	  = thread_pool<derived_type, indices>;
		using memory_buffer_t = memory_buffer<config>;

		thread_pool_t thread_pool_val{};

		RT_TM_INLINE device() noexcept						   = default;
		RT_TM_INLINE device& operator=(const device&) noexcept = delete;
		RT_TM_INLINE device(const device&) noexcept			   = delete;

		RT_TM_FORCE_INLINE device(size_t thread_count) : thread_pool_val{ thread_count } {};

		RT_TM_FORCE_INLINE void schedule_execution() {
			thread_pool_val.schedule_execution();
		}

		RT_TM_FORCE_INLINE void execute_tasks() {
			thread_pool_val.execute_tasks();
		}

		RT_TM_FORCE_INLINE void reset_state() {
			thread_pool_val.reset_state();
		}
	};

	template<model_config config, typename derived_type, impl_indices indices> struct device_registry<device_type::cpu, derived_type, config, indices> {
		RT_TM_INLINE device_registry() noexcept									 = default;
		RT_TM_INLINE device_registry& operator=(const device_registry&) noexcept = delete;
		RT_TM_INLINE device_registry(const device_registry&) noexcept			 = delete;

		RT_TM_FORCE_INLINE device_registry(size_t thread_count) {
			devices.emplace_back(std::make_unique<device<device_type::cpu, derived_type, config, indices>>(thread_count));
		}

		RT_TM_FORCE_INLINE auto& get_devices() {
			return devices;
		}

		std::vector<std::unique_ptr<device<device_type::cpu, derived_type, config, indices>>> devices{};
	};

}