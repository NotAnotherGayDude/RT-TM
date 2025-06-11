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

#include <rt_tm/common/monolithic_dispatcher.hpp>
#include <rt_tm/cpu/cpu_scheduler.hpp>
#include <rt_tm/common/common.hpp>
#include <rt_tm/common/tuple.hpp>
#include <thread>
#include <latch>

namespace rt_tm {

	RT_TM_FORCE_INLINE bool pin_thread_to_core(int core_id) {
#if defined(RT_TM_PLATFORM_WINDOWS)
		DWORD_PTR mask	 = 1ULL << core_id;
		HANDLE thread	 = GetCurrentThread();
		DWORD_PTR result = SetThreadAffinityMask(thread, mask);
		if (result == 0) {
			std::cerr << "Failed to set thread affinity on Windows. Error: " << GetLastError() << std::endl;
			return false;
		}
		return true;

#elif defined(RT_TM_PLATFORM_LINUX)
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(core_id, &cpuset);

		pthread_t current_thread = pthread_self();
		int result				 = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
		if (result != 0) {
			std::cerr << "Failed to set thread affinity on Linux. Error: " << result << std::endl;
			return false;
		}
		return true;

#elif defined(RT_TM_PLATFORM_MAC)
		thread_port_t thread				 = mach_thread_self();
		thread_affinity_policy_data_t policy = { core_id };
		kern_return_t result				 = thread_policy_set(thread, THREAD_AFFINITY_POLICY, ( thread_policy_t )&policy, 1);
		mach_port_deallocate(mach_task_self(), thread);
		if (result != KERN_SUCCESS) {
			std::cerr << "Failed to set thread affinity on macOS. Error: " << result << std::endl;
			return false;
		}
		return true;

#else
		std::cerr << "Thread pinning is not supported on this platform." << std::endl;
		return false;
#endif
	}

	RT_TM_FORCE_INLINE void raise_current_thread_priority() {
#if defined(RT_TM_PLATFORM_WINDOWS)
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST)) {
			std::cerr << "Failed to set thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif defined(RT_TM_PLATFORM_LINUX) || defined(RT_TM_PLATFORM_MAC)
		pthread_t this_thread = pthread_self();

		sched_param sch_params;
		sch_params.sched_priority = 0;

		int policy;
		if (pthread_getschedparam(this_thread, &policy, &sch_params) != 0) {
			std::cerr << "Failed to get thread sched param: " << strerror(errno) << std::endl;
			return;
		}

		int max_priority = sched_get_priority_max(policy);
		if (max_priority == -1) {
			std::cerr << "Failed to get max thread priority: " << strerror(errno) << std::endl;
			return;
		}

		sch_params.sched_priority = max_priority;

		if (pthread_setschedparam(this_thread, policy, &sch_params) != 0) {
			std::cerr << "Failed to set thread priority: " << strerror(errno) << std::endl;
		}
#else
	#warning "Thread priority adjustment not supported on this platform."
#endif
	}

	RT_TM_FORCE_INLINE void reset_current_thread_priority() {
#if defined(RT_TM_PLATFORM_WINDOWS)
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_NORMAL)) {
			std::cerr << "Failed to reset thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif defined(RT_TM_PLATFORM_LINUX) || defined(RT_TM_PLATFORM_MAC)
		pthread_t this_thread = pthread_self();

		sched_param sch_params;
		int policy;
		if (pthread_getschedparam(this_thread, &policy, &sch_params) != 0) {
			std::cerr << "Failed to get thread sched param: " << strerror(errno) << std::endl;
			return;
		}

		int min_priority = sched_get_priority_min(policy);
		int max_priority = sched_get_priority_max(policy);
		if (min_priority == -1 || max_priority == -1) {
			std::cerr << "Failed to get min/max priority: " << strerror(errno) << std::endl;
			return;
		}

		sch_params.sched_priority = (min_priority + max_priority) / 2;

		if (pthread_setschedparam(this_thread, policy, &sch_params) != 0) {
			std::cerr << "Failed to reset thread priority: " << strerror(errno) << std::endl;
		}
#else
	#warning "Thread priority adjustment not supported on this platform."
#endif
	}

	template<typename derived_type, impl_indices indices> struct thread_pool {
		RT_TM_FORCE_INLINE thread_pool() noexcept							   = delete;
		RT_TM_FORCE_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
		RT_TM_FORCE_INLINE thread_pool(const thread_pool&) noexcept			   = delete;

		RT_TM_FORCE_INLINE thread_pool(size_t thread_count_new) {
			worker_latches.resize(thread_count_new);
			threads.resize(thread_count_new);
			main_thread_latch = std::make_unique<std::latch>(static_cast<ptrdiff_t>(thread_count_new));
			for (size_t x = 0; x < thread_count_new; ++x) {
				worker_latches[x] = std::make_unique<std::latch>(static_cast<ptrdiff_t>(1));
				threads[x]		  = std::thread{ [&, x] {
					if (x < (thread_count_new % 3) == 0) {
						thread_function_impl<true>(x);
					} else {
						thread_function_impl<false>(x);
					}
				} };
			}
		}

		template<bool raise_priority> RT_TM_FORCE_INLINE void thread_function_impl(size_t thread_index) {
			if (thread_index % 2 == 0) {
				pin_thread_to_core(thread_index % 2);
			}
			while (!stop.load(std::memory_order_acquire)) {
				worker_latches[thread_index]->wait();
				if constexpr (raise_priority) {
					raise_current_thread_priority();
				}
				if (!stop.load(std::memory_order_acquire)) {
					static_cast<derived_type*>(this)->impl<thread_function>(thread_index);
				}
				if constexpr (raise_priority) {
					reset_current_thread_priority();
				}
				if (!main_thread_latch->try_wait()) {
					main_thread_latch->count_down();
				}
				worker_latches[thread_index] = std::make_unique<std::latch>(static_cast<ptrdiff_t>(1));
			}
		}

		RT_TM_FORCE_INLINE void execute_tasks() {
			main_thread_latch = std::make_unique<std::latch>(static_cast<ptrdiff_t>(threads.size()));
			for (auto& value: worker_latches) {
				if (!value->try_wait()) {
					value->count_down();
				}
			}
			main_thread_latch->wait();
		}

		RT_TM_FORCE_INLINE ~thread_pool() {
			stop.store(true, std::memory_order_release);
			for (auto& value: worker_latches) {
				if (!value->try_wait()) {
					value->count_down();
				}
			}
			for (auto& value: threads) {
				if (value.joinable()) {
					value.join();
				}
			}
		};

	  protected:
		std::vector<std::unique_ptr<std::latch>> worker_latches{};
		std::unique_ptr<std::latch> main_thread_latch{};
		std::vector<std::thread> threads{};
		std::atomic_bool stop{};
	};

}