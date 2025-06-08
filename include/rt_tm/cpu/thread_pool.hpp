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
#include <rt_tm/cpu/cpu_op_core.hpp>
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

	template<typename index_sequence> struct op_holder;

	template<size_t... inputs> struct op_holder<std::index_sequence<inputs...>> {
		using tuple_thread_t = tuple<std::vector<cpu_op_core_thread<inputs + 1>>...>;
		using tuple_core_t	 = tuple<std::vector<cpu_op_core<inputs + 1>>...>;
		tuple_thread_t thread_ops{};
		tuple_core_t core_ops{};

		template<size_t current_index = 0> RT_TM_FORCE_INLINE cpu_op_core_thread_base* emplace_back_core_thread(size_t index) {
			if constexpr (current_index < sizeof...(inputs)) {
				if (current_index == index) {
					auto& vec									   = get<current_index>(thread_ops);
					cpu_op_core_thread<current_index + 1>& new_ref = vec.emplace_back();
					return static_cast<cpu_op_core_thread_base*>(&new_ref);
				} else {
					return emplace_back_core_thread<current_index + 1>(index);
				}
			} else {
				return nullptr;
			}
		}

		template<size_t current_index = 0> RT_TM_FORCE_INLINE cpu_op_core_thread_base* get_core_thread(size_t input_count, size_t index) {
			if constexpr (current_index < sizeof...(inputs)) {
				if (current_index == input_count) {
					auto& vec									   = get<current_index>(thread_ops);
					cpu_op_core_thread<current_index + 1>& new_ref = vec[index];
					return static_cast<cpu_op_core_thread_base*>(&new_ref);
				} else {
					return get_core_thread<current_index + 1>(input_count, index);
				}
			} else {
				return nullptr;
			}
		}

		template<size_t current_index = 0> RT_TM_FORCE_INLINE cpu_op_core_base* emplace_back_core(size_t index) {
			if constexpr (current_index < sizeof...(inputs)) {
				if (current_index == index) {
					auto& vec								= get<current_index>(core_ops);
					cpu_op_core<current_index + 1>& new_ref = vec.emplace_back();
					return static_cast<cpu_op_core_base*>(&new_ref);
				} else {
					return emplace_back_core<current_index + 1>(index);
				}
			}
			return nullptr;
		}

		template<size_t current_index = 0> RT_TM_FORCE_INLINE cpu_op_core_base* get_core(size_t input_count, size_t index) {
			if constexpr (current_index < sizeof...(inputs)) {
				if (current_index == input_count) {
					auto& vec								= get<current_index>(core_ops);
					cpu_op_core<current_index + 1>& new_ref = vec[index];
					return static_cast<cpu_op_core_base*>(&new_ref);
				} else {
					return get_core<current_index + 1>(input_count, index);
				}
			} else {
				return nullptr;
			}
		}
	};

	template<size_t max_inputs, impl_indices indices, size_t depth_new> struct scheduler_depth : public cpu_scheduler<scheduler_depth<max_inputs, indices, depth_new>>,
																								 public op_holder<std::make_index_sequence<max_inputs>> {
		static constexpr size_t depth{ depth_new };
		RT_TM_FORCE_INLINE scheduler_depth(size_t max_depth, size_t thread_count) {};
		//RT_TM_FORCE_INLINE scheduler_depth() noexcept								   = default;
		RT_TM_FORCE_INLINE scheduler_depth& operator=(const scheduler_depth&) noexcept = delete;
		RT_TM_FORCE_INLINE scheduler_depth(const scheduler_depth&) noexcept			   = delete;

		RT_TM_FORCE_INLINE bool schedule_execution(std::vector<core_base*>& ops) {
			if (depth < max_depth) {
				this->emplace_back_core(2);
				this->emplace_back_core_thread(2)->thread_index;
				this->get_core_thread(2, 2);
				this->get_core(2, 2);
				end_latch	= std::make_unique<std::latch>(static_cast<ptrdiff_t>(op_threads.size()));
				start_latch = std::make_unique<std::latch>(static_cast<ptrdiff_t>(op_threads.size()));
				for (size_t x = 0; x < op_thread_chains.size(); ++x) {
					op_thread_chains[x].end_latch	= end_latch.get();
					op_thread_chains[x].start_latch = start_latch.get();
				}
				return true;
			}
			return false;
		}

		RT_TM_FORCE_INLINE bool reset_state() {
			if (depth < max_depth) {
				end_latch	= std::make_unique<std::latch>(static_cast<ptrdiff_t>(op_threads.size()));
				start_latch = std::make_unique<std::latch>(static_cast<ptrdiff_t>(op_threads.size()));
				for (size_t x = 0; x < op_thread_chains.size(); ++x) {
					op_thread_chains[x].end_latch	= end_latch.get();
					op_thread_chains[x].start_latch = start_latch.get();
				}
				return true;
			} else {
				return false;
			}
		}

		RT_TM_FORCE_INLINE bool thread_function(size_t thread_index) {
			if (depth < max_depth) {
				cpu_op_core_thread_chain& op_chain{ op_thread_chains[thread_index] };
				for (size_t x = 0; x < op_chain.cpu_op_core_thread_ptrs.size() - 1; ++x) {
					cpu_op_core_thread_base* new_task{ static_cast<cpu_op_core_thread_base*>(op_thread_chains[thread_index].cpu_op_core_thread_ptrs[x]) };
					new_task->thread_count;
					//op_dispatcher_final<rt_tm::device_type::cpu, indices>::impl(new_task);
				}
				op_chain.start_latch->arrive_and_wait();
				//op_dispatcher_final<rt_tm::device_type::cpu, indices>::impl(op_chain.cpu_op_core_thread_ptrs[op_chain.cpu_op_core_thread_ptrs.size() - 1]);
				op_chain.end_latch->arrive_and_wait();
				return true;
			} else {
				return false;
			}
		}

	  protected:
		std::vector<cpu_op_core_thread_chain> op_thread_chains{};
		std::vector<cpu_op_core_thread_base*> op_threads{};
		std::vector<cpu_op_core_base*> op_bases{};
		std::unique_ptr<std::latch> start_latch{};
		std::unique_ptr<std::latch> end_latch{};
		size_t thread_count{};
		size_t max_depth{};
	};

	template<typename... bases> struct scheduler_depths : public bases... {
		RT_TM_FORCE_INLINE scheduler_depths(size_t max_depth, size_t thread_count) : bases{ max_depth, thread_count }... {};
		RT_TM_FORCE_INLINE scheduler_depths& operator=(scheduler_depths&&)		= delete;
		RT_TM_FORCE_INLINE scheduler_depths(scheduler_depths&&)					= delete;
		RT_TM_FORCE_INLINE scheduler_depths& operator=(const scheduler_depths&) = delete;
		RT_TM_FORCE_INLINE scheduler_depths(const scheduler_depths&)			= delete;
		template<typename op_entity_type, typename... arg_types> RT_TM_FORCE_INLINE bool iterate_values_impl(arg_types&&... args) {
			return op_entity_type::thread_function(std::forward<arg_types>(args)...);
		}

		template<typename... arg_types> RT_TM_FORCE_INLINE void iterate_values(arg_types&&... args) {
			(iterate_values_impl<bases>(args...) && ...);
		}

		template<typename op_entity_type, typename... arg_types> RT_TM_FORCE_INLINE bool reset_state_impl(arg_types&&... args) {
			return op_entity_type::reset_state(std::forward<arg_types>(args)...);
		}

		template<typename... arg_types> RT_TM_FORCE_INLINE void reset_state(arg_types&&... args) {
			(reset_state_impl<bases>(args...) && ...);
		}

		template<typename op_entity_type, typename... arg_types> RT_TM_FORCE_INLINE bool schedule_execution_impl(arg_types&&... args) {
			return op_entity_type::schedule_execution(std::forward<arg_types>(args)...);
		}

		template<typename... arg_types> RT_TM_FORCE_INLINE void schedule_execution(arg_types&&... args) {
			(schedule_execution_impl<bases>(args...) && ...);
		}
	};

	template<size_t max_inputs, impl_indices indices, typename index_sequence> struct get_scheduler_base;

	template<size_t max_inputs, impl_indices indices, size_t... index> struct get_scheduler_base<max_inputs, indices, std::index_sequence<index...>> {
		using type = scheduler_depths<scheduler_depth<max_inputs, indices, index>...>;
	};

	template<size_t max_inputs, impl_indices indices> using scheduler_base_t = typename get_scheduler_base<max_inputs, indices, std::make_index_sequence<64>>::type;

	template<size_t max_inputs, impl_indices indices> struct thread_pool : public scheduler_base_t<max_inputs, indices> {
		RT_TM_FORCE_INLINE thread_pool() noexcept							   = delete;
		RT_TM_FORCE_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
		RT_TM_FORCE_INLINE thread_pool(const thread_pool&) noexcept			   = delete;

		RT_TM_FORCE_INLINE thread_pool(size_t thread_count_new) : scheduler_base_t<max_inputs, indices>{ thread_count_new, thread_count_new } {
			worker_latches.resize(thread_count_new);
			threads.resize(thread_count_new);
			main_thread_latch = std::make_unique<std::latch>(static_cast<ptrdiff_t>(thread_count_new));
			for (size_t x = 0; x < thread_count_new; ++x) {
				worker_latches[x] = std::make_unique<std::latch>(static_cast<ptrdiff_t>(1));
				threads[x]		  = std::thread{ [&, x] {
					if (x < (thread_count_new % 3) == 0) {
						thread_function<true>(x);
					} else {
						thread_function<false>(x);
					}
				} };
			}
		}

		template<bool raise_priority> RT_TM_FORCE_INLINE void thread_function(size_t thread_index) {
			if (thread_index % 2 == 0) {
				pin_thread_to_core(thread_index % 2);
			}
			while (!stop.load(std::memory_order_acquire)) {
				worker_latches[thread_index]->wait();
				if constexpr (raise_priority) {
					raise_current_thread_priority();
				}
				this->iterate_values(threads.size());
				if constexpr (raise_priority) {
					reset_current_thread_priority();
				}
				main_thread_latch->count_down();
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