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

#if defined(RT_TM_AVX2)

namespace rt_tm {

	template<uint64_t cpu_arch_index, kernel_type type, typename transform_type, typename... operand_types> struct kernel_dispatcher_impl;

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::get_rows, transform_type, float, block_q8_0<half>, int32_t> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const block_q8_0<half>*, const int32_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::get_rows, transform_type, float, float, int32_t> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const int32_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::rms_norm, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::transpose, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::view, transform_type, int16_t, int16_t> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, int16_t*, const int16_t*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::mul, transform_type, float, float, block_q8_0<half>> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const block_q8_0<half>*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::mul_mat, transform_type, float, block_q8_0<half>, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const block_q8_0<half>*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::mul_mat, transform_type, float, int16_t, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const int16_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::rope, transform_type, float, float, int32_t, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const int32_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::copy, transform_type, int16_t, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, int16_t*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::permute, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::mul_mat, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::softmax, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static __m256 _mm256_exp_ps(__m256 invec) {
			alignas(32) float element[8];
			_mm256_store_ps(element, invec);
			return _mm256_setr_ps(expf(element[0]), expf(element[1]), expf(element[2]), expf(element[3]), expf(element[4]), expf(element[5]), expf(element[6]), expf(element[7]));
		}

		RT_TM_FORCE_INLINE static float horizontal_max(__m256 vec) {
			__m128 low	  = _mm256_castps256_ps128(vec);
			__m128 high	  = _mm256_extractf128_ps(vec, 1);
			__m128 max128 = _mm_max_ps(low, high);

			__m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1));
			max128		= _mm_max_ps(max128, shuf);
			shuf		= _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
			max128		= _mm_max_ps(max128, shuf);

			return _mm_cvtss_f32(max128);
		}

		RT_TM_FORCE_INLINE static float horizontal_sum(__m256 vec) {
			__m128 low	  = _mm256_castps256_ps128(vec);
			__m128 high	  = _mm256_extractf128_ps(vec, 1);
			__m128 sum128 = _mm_add_ps(low, high);

			__m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
			sum128		= _mm_add_ps(sum128, shuf);
			shuf		= _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
			sum128		= _mm_add_ps(sum128, shuf);

			return _mm_cvtss_f32(sum128);
		}

		RT_TM_FORCE_INLINE static void impl(uint64_t count, float* output, const float* input01, const float* input02) {
			/*
			const size_t simd_width = 8;
			const size_t simd_count = count / simd_width;
			const size_t remainder	= count % simd_width;

			__m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

			for (size_t i = 0; i < simd_count; ++i) {
				__m256 logits		 = _mm256_loadu_ps(input01 + i * simd_width);
				__m256 mask			 = _mm256_loadu_ps(input02 + i * simd_width);
				__m256 masked_logits = _mm256_blendv_ps(_mm256_set1_ps(-std::numeric_limits<float>::infinity()), logits, mask);
				max_vec				 = _mm256_max_ps(max_vec, masked_logits);
			}

			float max_val = horizontal_max(max_vec);

			for (size_t i = simd_count * simd_width; i < count; ++i) {
				if (input02[i] != 0.0f) {
					max_val = std::max(max_val, input01[i]);
				}
			}

			const __m256 max_broadcast = _mm256_set1_ps(max_val);
			__m256 sum_vec			   = _mm256_setzero_ps();

			for (size_t i = 0; i < simd_count; ++i) {
				__m256 logits	= _mm256_loadu_ps(input01 + i * simd_width);
				__m256 mask		= _mm256_loadu_ps(input02 + i * simd_width);
				__m256 shifted	= _mm256_sub_ps(logits, max_broadcast);
				__m256 exp_vals = _mm256_exp_ps(shifted);
				exp_vals		= _mm256_and_ps(exp_vals, mask);
				_mm256_storeu_ps(output + i * simd_width, exp_vals);
				sum_vec = _mm256_add_ps(sum_vec, exp_vals);
			}

			float total_sum = horizontal_sum(sum_vec);

			for (size_t i = simd_count * simd_width; i < count; ++i) {
				if (input02[i] != 0.0f) {
					float exp_val = std::exp(input01[i] - max_val);
					output[i]	  = exp_val;
					total_sum += exp_val;
				} else {
					output[i] = 0.0f;
				}
			}

			const __m256 inv_sum = _mm256_set1_ps(1.0f / total_sum);

			for (size_t i = 0; i < simd_count; ++i) {
				__m256 exp_vals = _mm256_loadu_ps(output + i * simd_width);
				_mm256_storeu_ps(output + i * simd_width, _mm256_mul_ps(exp_vals, inv_sum));
			}

			for (size_t i = simd_count * simd_width; i < count; ++i) {
				output[i] /= total_sum;
			}*/
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::add, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::silu, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::cont, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::reshape, transform_type, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*) {
		}
	};

	template<typename transform_type> struct kernel_dispatcher_impl<1, kernel_type::mul, transform_type, float, float, float> {
		RT_TM_FORCE_INLINE static void impl(uint64_t count, float*, const float*, const float*) {
		}
	};

};

#endif
