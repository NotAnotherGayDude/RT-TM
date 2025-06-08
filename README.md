# RT-TM: RealTime TensorMath

*A Type-Safe, Depth-Scheduled ML Runtime for Modern CPUs*

**One-liner vision:**

> ***A static-dynamic hybrid ML runtime that compiles your model graph into a lock-free, thread-coordinated execution plan with zero vtables, zero RTTI, and pure type safety — tuned for real-time CPU inference.***

---

**RT-TM** is a new kind of machine learning runtime:
It transforms models like Meta's LLaMA 3 into **fully type-specialized, depth-indexed op graphs** that execute using a custom-designed hybrid thread scheduler.
Each op is stored as a concrete type — no virtual dispatch — and is scheduled according to its depth and dataflow dependencies.

The result is a runtime that:

✅ ***Executes continuous op chains without synchronization overhead***  
✅ ***Synchronizes only at blocking ops (e.g. matrix multiplications)***  
✅ ***Uses a depth-indexed threadpool with per-thread op chains***  
✅ ***Has cache-optimized storage for all ops — no vtables***  
✅ ***Supports quantized types and arbitrary op type specializations***  
✅ ***Exposes a clean, pure C++ API***  

---

# 🖥️ **Sample Usage (from RT-TM API)**

```cpp
static constexpr rt_tm::model_config model_config{ .arch = rt_tm::model_arch::llama, .exceptions = false };

// Parse model from GGUF
auto model_graph = rt_tm::harbinger<model_config>::parse_model_graph<rt_tm::model_format::gguf>(argv[2]);

// Create op graph with thread configuration
rt_tm::op_graph_config graph_config{ .num_threads = 12 };
rt_tm::op_graph<model_config> op_graph{ rt_tm::harbinger<model_config>::create_op_graph(graph_config, model_graph) };

// Setup input stream
rt_tm::input_session_config session_config{ .stream = std::cin, .max_tokens = 1024 };
rt_tm::input_session input_session{ session_config };

// Main inference loop
while (input_session) {
    op_graph.process_input(input_session);
}
```

---

# 🗺️ **Execution Model Diagram**

```
Model Parsing (GGUF) → model_graph (static topology + metadata)
                          ↓
harbinger::create_op_graph() → op_graph<model_config>
                          ↓
op_graph {
    op_graph_bases {
        op_graph_bases_op<op_enum_value> → vector<typed core<T>>
    }
    device_registry {
        thread_pool {
            scheduler_depths {
                scheduler_depth<depth N> {
                    op_holder<input_count> {
                        vector<cpu_op_core_thread<input_count>>
                    }
                    thread-coordinated op_chain execution
                }
            }
        }
    }
}
                          ↓
process_input(input_session):
    → resets state
    → schedules execution
    → executes tasks across thread pool
```

---

# 💡 **Key Innovations**

* **Static-dynamic hybrid execution:** compile-time typed op graph + runtime depth scheduling  
* **Per-depth continuous-op bursting:** threads blast forward until sync barrier required  
* **Zero vtables / RTTI:** all ops are concrete types, not heap-allocated polymorphic instances  
* **Fast dispatch:** compile-time generated dispatch chains per op type / type group  
* **Extensible:** easily add support for quantized ops, arch-specific kernels (AVX, AVX512, SVE)  

---

# 🚀 **Current Status**

✅ Core architecture implemented  
✅ Working op graph execution  
✅ Depth-aware thread scheduling proven  
✅ LLaMA 3 GGUF parsing working  
✅ Static op graph storage system built  

Coming next:

* Quantized op specialization  
* Memory arena  
* Profiler  
* Full LLaMA forward pass benchmarks  
* Public engine release  

---

# ✨ **Conclusion**

RT-TM represents an attempt to rethink how CPU-based ML runtimes are structured.
Instead of treating execution as a stream of generic ops with runtime indirection, RT-TM fully embraces **C++ static typing and compile-time specialization**, turning ML models into highly optimized, depth-aware execution pipelines that can run in real time on modern CPUs.

---