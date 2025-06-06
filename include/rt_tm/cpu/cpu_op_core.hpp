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

#include <rt_tm/common/core_base.hpp>
#include <latch>

namespace rt_tm {

	struct cpu_op_core_input {
		RT_TM_FORCE_INLINE cpu_op_core_input() noexcept = default;
		RT_TM_FORCE_INLINE cpu_op_core_input(core_base& other) : core_base_ptr{ &other } {};
		const core_base* core_base_ptr{};
	};

	struct cpu_op_core_output {
		RT_TM_FORCE_INLINE cpu_op_core_output() noexcept = default;
		RT_TM_FORCE_INLINE cpu_op_core_output(core_base& other) : core_base_ptr{ &other } {};
		core_base* core_base_ptr{};
	};

	struct cpu_op_core_base {
		RT_TM_FORCE_INLINE cpu_op_core_base() = default;
		RT_TM_FORCE_INLINE cpu_op_core_base(core_base&) {};
		core_base* core_base_ptr{};

	  protected:
		~cpu_op_core_base() noexcept = default;
	};

	template<size_t input_count> struct cpu_op_core : public cpu_op_core_base {
		RT_TM_FORCE_INLINE cpu_op_core() noexcept = default;
		RT_TM_FORCE_INLINE cpu_op_core(core_base&) {};
		cpu_op_core_input input[input_count]{};
		cpu_op_core_output output{};
	};

	struct cpu_op_core_thread_base {
		core_base* core_base_ptr{};
		size_t thread_index{};
		size_t thread_count{};

	  protected:
		~cpu_op_core_thread_base() noexcept = default;
	};

	template<size_t input_count> struct cpu_op_core_thread : public cpu_op_core_thread_base {
		std::vector<const void*> inputs[input_count]{};
		cpu_op_core<input_count>* cpu_op_core_ptr{};
		std::vector<void*> output{};
	};

	// Make the "non-continuous-op" the last one in this vector!
	struct cpu_op_core_thread_chain {
		std::vector<cpu_op_core_thread_base*> cpu_op_core_thread_ptrs{};
		std::latch* start_latch{};
		std::latch* end_latch{};
	};

}