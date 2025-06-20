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

#include <cstdint>

namespace nihilus {

	inline static constexpr uint64_t Q_SIZE{ 32 };

	using half	   = int16_t;
	using half2	   = int32_t;
	using fp16_t   = int16_t;
	using bf16_t   = int16_t;
	using float_32 = float;
	using float_64 = double;

	template<typename half_type> struct block_q8_0 {
		half_type d;
		int8_t qs[Q_SIZE];
	};
	static_assert(sizeof(block_q8_0<half>) == sizeof(half) + Q_SIZE, "Wrong q8_0 block size/padding.");

}
