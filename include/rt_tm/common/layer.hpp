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

#include <rt_tm/common/config.hpp>
#include <rt_tm/common/array.hpp>

namespace rt_tm {

	enum class llama_layer_type : uint64_t {
		token_embedding,// Invariant: [vocab_size, embedding_dim]
		attention_block,// Invariant: ALL ops share (embedding_dim, head_count) basis
		ffn_block,// Invariant: ALL ops share (embedding_dim, ff_multiplier) basis
		rms_norm,// Invariant: [*, embedding_dim] → [*, embedding_dim]
		residual,// Invariant: same shape in = same shape out
		lm_head,// Invariant: [embedding_dim, vocab_size]
		count
	};

	template<model_arch arch, typename layer_type, layer_type type> struct layer;

	template<> struct layer<model_arch::llama, llama_layer_type, llama_layer_type::token_embedding> {};
	template<> struct layer<model_arch::llama, llama_layer_type, llama_layer_type::attention_block> {};
	template<> struct layer<model_arch::llama, llama_layer_type, llama_layer_type::ffn_block> {};
	template<> struct layer<model_arch::llama, llama_layer_type, llama_layer_type::rms_norm> {};
	template<> struct layer<model_arch::llama, llama_layer_type, llama_layer_type::residual> {};
	template<> struct layer<model_arch::llama, llama_layer_type, llama_layer_type::lm_head> {};

	template<model_arch arch, llama_layer_type layer_type> struct layer_traits_impl;

	template<model_arch arch, llama_layer_type layer_type> struct layer_traits;

	template<> struct layer_traits_impl<model_arch::llama, llama_layer_type::token_embedding> {
		size_t embedding_dim{};
		size_t vocab_size{};
	};

	template<> struct layer_traits<model_arch::llama, llama_layer_type::token_embedding> {
		static constexpr array vocab_sizes{
			32000ull,// v1/v2 models
			128256ull,// v3 models
		};

		static constexpr array embedding_dims{
			2048ull,// 1B
			3072ull,// 3B
			4096ull,// 7B
			5120ull,// 13B
			6656ull,// 30B
			8192ull,// 70B
			16384ull,// 405B
		};

		enum class token_embedding_config : size_t {
			llama_1b_v1	  = 0,
			llama_3b_v1	  = 1,
			llama_7b_v1	  = 2,
			llama_13b_v1  = 3,
			llama_30b_v1  = 4,
			llama_70b_v1  = 5,
			llama_405b_v1 = 6,
			llama_1b_v3	  = 7,
			llama_3b_v3	  = 8,
			llama_7b_v3	  = 9,
			llama_13b_v3  = 10,
			llama_30b_v3  = 11,
			llama_70b_v3  = 12,
			llama_405b_v3 = 13,
			count		  = vocab_sizes.size() * embedding_dims.size()
		};

		static constexpr array dims{ [] {
			array<layer_traits_impl<model_arch::llama, llama_layer_type::token_embedding>, embedding_dims.size() * vocab_sizes.size()> return_values{};
			for (size_t x = 0; x < vocab_sizes.size(); ++x) {
				for (size_t y = 0; y < embedding_dims.size(); ++y) {
					return_values[x * embedding_dims.size() + y] = layer_traits_impl<model_arch::llama, llama_layer_type::token_embedding>{ embedding_dims[y], vocab_sizes[x] };
				}
			}
			return return_values;
		}() };
	};

	template<> struct layer_traits_impl<model_arch::llama, llama_layer_type::attention_block> {
		size_t embedding_dim{};
		size_t head_count{};
		mutable size_t sequence_length{};
	};

	template<> struct layer_traits<model_arch::llama, llama_layer_type::attention_block> {
		static constexpr array embedding_dims{
			2048ull,// 1B
			3072ull,// 3B
			4096ull,// 7B
			5120ull,// 13B
			8192ull,// 70B
			16384ull,// 405B
		};

		static constexpr array head_counts{
			32ull,// 1B: 2048/32 = 64 head_dim (non-standard!)
			24ull,// 3B: 3072/24 = 128 head_dim ✅
			32ull,// 7B: 4096/32 = 128 head_dim ✅
			40ull,// 13B: 5120/40 = 128 head_dim ✅
			64ull,// 70B: 8192/64 = 128 head_dim ✅
			128ull,// 405B: 16384/128 = 128 head_dim ✅
		};

		enum class attention_block_config : size_t {
			embed_2048_heads_32	  = 0,// 1B
			embed_3072_heads_24	  = 1,// 3B
			embed_4096_heads_32	  = 2,// 7B
			embed_5120_heads_40	  = 3,// 13B
			embed_8192_heads_64	  = 4,// 70B
			embed_16384_heads_128 = 5,// 405B
			count				  = 6,
		};

		static constexpr array dims{ [] {
			array<layer_traits_impl<model_arch::llama, llama_layer_type::attention_block>, 6> return_values{};
			for (size_t i = 0; i < 6; ++i) {
				return_values[i] = { embedding_dims[i], head_counts[i], 0 };
			}
			return return_values;
		}() };
	};

	template<> struct layer_traits_impl<model_arch::llama, llama_layer_type::ffn_block> {
		size_t embedding_dim{};
		size_t feed_forward_length{};
	};

	template<> struct layer_traits<model_arch::llama, llama_layer_type::ffn_block> {
		static constexpr array embedding_dims{
			2048ull,// 1B
			3072ull,// 3B
			4096ull,// 7B
			5120ull,// 13B
			8192ull,// 70B
			16384ull,// 405B
		};

		static constexpr array feed_forward_lengths{
			8192ull,// 1B:  8192 / 2048  = 4.0x
			8192ull,// 3B:  8192 / 3072  = 2.67x
			11008ull,// 7B:  11008 / 4096 = 2.69x
			13824ull,// 13B: 13824 / 5120 = 2.7x
			28672ull,// 70B: 28672 / 8192 = 3.5x
			53248ull,// 405B: 53248 / 16384 = 3.25x
		};

		enum class ffn_block_config : size_t {
			embed_2048_ff_8192	 = 0,// 1B
			embed_3072_ff_8192	 = 1,// 3B
			embed_4096_ff_11008	 = 2,// 7B
			embed_5120_ff_13824	 = 3,// 13B
			embed_8192_ff_28672	 = 4,// 70B
			embed_16384_ff_53248 = 5,// 405B
			count				 = 6,
		};

		static constexpr array dims{ [] {
			array<layer_traits_impl<model_arch::llama, llama_layer_type::ffn_block>, 6> return_values{};
			for (size_t i = 0; i < 6; ++i) {
				return_values[i] = { embedding_dims[i], feed_forward_lengths[i] };
			}
			return return_values;
		}() };
	};

	template<> struct layer_traits_impl<model_arch::llama, llama_layer_type::rms_norm> {
		size_t embedding_dim{};
	};

	template<> struct layer_traits<model_arch::llama, llama_layer_type::rms_norm> {
		static constexpr array embedding_dims{
			2048ull,// 1B
			3072ull,// 3B
			4096ull,// 7B
			5120ull,// 13B
			8192ull,// 70B
			16384ull,// 405B
		};

		enum class rms_norm_config : size_t {
			embed_2048	= 0,// 1B
			embed_3072	= 1,// 3B
			embed_4096	= 2,// 7B
			embed_5120	= 3,// 13B
			embed_8192	= 4,// 70B
			embed_16384 = 5,// 405B
			count		= 6,
		};

		static constexpr array dims{ [] {
			array<layer_traits_impl<model_arch::llama, llama_layer_type::rms_norm>, 6> return_values{};
			for (size_t i = 0; i < 6; ++i) {
				return_values[i] = { embedding_dims[i] };
			}
			return return_values;
		}() };
	};

	template<> struct layer_traits_impl<model_arch::llama, llama_layer_type::residual> {
		size_t embedding_dim{};// ✅ For tensor shape validation/allocation
	};

	template<> struct layer_traits<model_arch::llama, llama_layer_type::residual> {
		static constexpr array embedding_dims{
			2048ull,// 1B
			3072ull,// 3B
			4096ull,// 7B
			5120ull,// 13B
			8192ull,// 70B
			16384ull,// 405B
		};

		enum class residual_config : size_t {
			embed_2048	= 0,// 1B
			embed_3072	= 1,// 3B
			embed_4096	= 2,// 7B
			embed_5120	= 3,// 13B
			embed_8192	= 4,// 70B
			embed_16384 = 5,// 405B
			count		= 6,
		};

		static constexpr array dims{ [] {
			array<layer_traits_impl<model_arch::llama, llama_layer_type::residual>, 6> return_values{};
			for (size_t i = 0; i < 6; ++i) {
				return_values[i] = { embedding_dims[i] };
			}
			return return_values;
		}() };
	};

	template<> struct layer_traits_impl<model_arch::llama, llama_layer_type::lm_head> {
		size_t embedding_dim{};// ✅ Input dimension
		size_t vocab_size{};// ✅ Output dimension (missing!)
	};

	template<> struct layer_traits<model_arch::llama, llama_layer_type::lm_head> {
		static constexpr array vocab_sizes{
			32000ull,// v1/v2 models
			128256ull,// v3 models
		};

		static constexpr array embedding_dims{
			2048ull,// 1B
			3072ull,// 3B
			4096ull,// 7B
			5120ull,// 13B
			6656ull,// 30B
			8192ull,// 70B
			16384ull,// 405B
		};

		enum class lm_head_config : size_t {
			llama_1b_v1	  = 0,
			llama_3b_v1	  = 1,
			llama_7b_v1	  = 2,
			llama_13b_v1  = 3,
			llama_30b_v1  = 4,
			llama_70b_v1  = 5,
			llama_405b_v1 = 6,
			llama_1b_v3	  = 7,
			llama_3b_v3	  = 8,
			llama_7b_v3	  = 9,
			llama_13b_v3  = 10,
			llama_30b_v3  = 11,
			llama_70b_v3  = 12,
			llama_405b_v3 = 13,
			count		  = vocab_sizes.size() * embedding_dims.size()
		};

		static constexpr array dims{ [] {
			array<layer_traits_impl<model_arch::llama, llama_layer_type::lm_head>, embedding_dims.size() * vocab_sizes.size()> return_values{};
			for (size_t x = 0; x < vocab_sizes.size(); ++x) {
				for (size_t y = 0; y < embedding_dims.size(); ++y) {
					return_values[x * embedding_dims.size() + y] = { embedding_dims[y], vocab_sizes[x] };
				}
			}
			return return_values;
		}() };
	};
}
