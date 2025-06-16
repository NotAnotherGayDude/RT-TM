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

#include <nihilus/cpu/simd/avx_2.hpp>
#include <nihilus/cpu/simd/avx_512.hpp>

#include <nihilus/cpu/simd/arm_neon.hpp>
#include <nihilus/cpu/simd/arm_sve2.hpp>

namespace nihilus {

	template<uint64_t cpu_arch_index, auto op_type, typename transform_type, typename... operand_types> struct kernel_dispatcher_impl;

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::inp_embd, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& token_embd_weight,
			const typename core_type::input_type02& inp_tokens) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::norm, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& input) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::attn_norm, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& norm_input,
			const typename core_type::input_type02& attn_norm_weight) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::qcur, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& attn_q_weight,
			const typename core_type::input_type02& attn_norm) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::qcur_reshaped, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& qcur) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::qcur_rope, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& qcur_reshaped,
			const typename core_type::input_type02& inp_pos, const typename core_type::input_type03& rope_freqs_weight) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kcur, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& attn_k_weight,
			const typename core_type::input_type02& attn_norm) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kcur_reshaped, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kcur) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kcur_rope, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kcur_reshaped,
			const typename core_type::input_type02& inp_pos, const typename core_type::input_type03& rope_freqs_weight) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::vcur, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& attn_v_weight,
			const typename core_type::input_type02& attn_norm) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::k_cache_view, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& cache_k) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::k_cache_view_copy, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kcur_rope) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::vcur_transposed, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& vcur) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::v_cache_view, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& cache_v) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::v_cache_view_copy, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& vcur_transposed) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::v, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& cache_v) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::k, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& cache_k) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::q, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& qcur_rope) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kq, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& k, const typename core_type::input_type02& q) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kq_soft_max, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kq,
			const typename core_type::input_type02& kq_mask) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kqv, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& v,
			const typename core_type::input_type02& kq_soft_max) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kqv_merged, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kqv) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kqv_merged_cont, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kqv_merged) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::kqv_out, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& attn_output_weight,
			const typename core_type::input_type02& kqv_merged_cont) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_inp, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kqv_out,
			const typename core_type::input_type02& l_out) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::norm_out, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_inp) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_norm, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& norm_out,
			const typename core_type::input_type02& ffn_norm_weight) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_gate, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_gate_weight,
			const typename core_type::input_type02& ffn_norm) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_silu, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_gate) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_up, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_up_weight,
			const typename core_type::input_type02& ffn_norm) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_gate_par, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_silu,
			const typename core_type::input_type02& ffn_up) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::ffn_out, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_down_weight,
			const typename core_type::input_type02& ffn_gate_par) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::l_out, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& ffn_out,
			const typename core_type::input_type02& ffn_inp) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::attn_residual, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& kqv_out,
			const typename core_type::input_type02& inp_out_ids) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::prev_residual, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& l_out,
			const typename core_type::input_type02& inp_out_ids) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::final_norm, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& l_out) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::result_norm, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& final_norm,
			const typename core_type::input_type02& output_norm_weight) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, llama_op_types::result_output, transform_type, core_type> {
		NIHILUS_FORCE_INLINE static void impl(size_t thread_index, size_t thread_count, core_type& output, const typename core_type::input_type01& output_weight,
			const typename core_type::input_type02& result_norm) {
		}
	};
}