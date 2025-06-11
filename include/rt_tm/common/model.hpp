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

#include <rt_tm/common/arch_traits.hpp>
#include <rt_tm/common/model_traits.hpp>
#include <rt_tm/cpu/thread_pool.hpp>
#include <rt_tm/common/h_params.hpp>
#include <rt_tm/common/tuple.hpp>

namespace rt_tm {

	RT_TM_FORCE_INLINE consteval auto generate_model_config(auto model_generation, auto model_size, kernel_type_profile kernel_profile, model_arch arch, bool exceptions = false,
		kv_cache_strategy cache_strategy = kv_cache_strategy::paged, bool use_gradient_checkpointing = false, rope_scaling_type rope_scaling = rope_scaling_type::linear,
		bool use_rotary_embeddings = true, size_t kv_cache_block_size = 16, bool use_flash_attention = true, norm_type rms_norm_type = norm_type::rms_standard,
		float norm_epsilon = 1e-6f) {
		model_config<decltype(model_generation), decltype(model_size)> config{ model_generation, model_size, kernel_profile, arch, cache_strategy, use_gradient_checkpointing,
			rope_scaling, use_rotary_embeddings, kv_cache_block_size, use_flash_attention, rms_norm_type, norm_epsilon, exceptions };
		return config;
	};

	template<impl_indices indices, model_config config> struct model;

	template<typename model_generation_type, typename model_size_type> struct model_base {
		model_config<model_generation_type, model_size_type> config{};
		virtual void execute_model() = 0;
		virtual ~model_base()		 = default;
	};

	template<impl_indices indices, model_config config> struct model
		: public model_base<decltype(config.model_size), decltype(config.model_generation)>,
		  public get_core_traits_base_t<indices, typename op_type_type<config.arch>::type, model<indices, config>,
			  model_traits<config.arch, config.model_size, config.model_generation>, kernel_type_profile_traits<config.kernel_profile>>,
		  public thread_pool<indices, model<indices, config>, model_traits<config.arch, config.model_size, config.model_generation>,
			  kernel_type_profile_traits<config.kernel_profile>> {
		using core_bases_type				  = get_core_traits_base_t<indices, typename op_type_type<config.arch>::type, model<indices, config>,
							model_traits<config.arch, config.model_size, config.model_generation>, kernel_type_profile_traits<config.kernel_profile>>;
		using model_traits_type				  = model_traits<config.arch, config.model_size, config.model_generation>;
		using op_type_type						  = typename model_traits_type::op_type_type;
		using kernel_type_profile_traits_type = kernel_type_profile_traits<config.kernel_profile>;
		using base_type						  = model_base<decltype(config.model_size), decltype(config.model_generation)>;
		inline static constexpr size_t total_required_bytes{
			collect_required_bytes<indices, typename model_traits_type::op_type_type, model<indices, config>, model_traits_type, kernel_type_profile_traits_type>::impl()
		};
		RT_TM_FORCE_INLINE model& operator=(model&&)	  = delete;
		RT_TM_FORCE_INLINE model(model&&)				  = delete;
		RT_TM_FORCE_INLINE model& operator=(const model&) = delete;
		RT_TM_FORCE_INLINE model(const model&)			  = delete;
		RT_TM_FORCE_INLINE model(const std::string_view& path_to_model_file, size_t thread_count = std::thread::hardware_concurrency())
			: thread_pool<indices, model<indices, config>, model_traits<config.arch, config.model_size, config.model_generation>,
				  kernel_type_profile_traits<config.kernel_profile>>{ thread_count } {
			memory.init(total_required_bytes);
			core_bases_type::template impl<memory_mapper, indices>(memory);
			core_bases_type::template impl<execution_planner, indices>(thread_count);
		}

		template<op_type_type type> RT_TM_FORCE_INLINE auto& get_core() {
			return *static_cast<core_traits<indices, type, model, model_traits_type, kernel_type_profile_traits_type>*>(this);
		}

		RT_TM_FORCE_INLINE void execute_model() {
			this->execute_tasks();
			// Perform all of the necessary stuff to execute the model - along with all of the constexpr values stored globally inside the class LOL!.
			// Because we only pay the "virtual overhead @ the top here == totally negligible.
		};

	  protected:
		memory_buffer<config> memory{};
	};

}
