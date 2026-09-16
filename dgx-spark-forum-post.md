# DGPP: fast, GB10-native inference on 1 to 4 DGX Sparks

Hi folks,

I have been building an inference engine called **DGPP** for NVIDIA DGX Spark. It is open source, written in C++ and CUDA, and focused entirely on getting strong real-world performance from GB10.

The project is here: [github.com/HawkBearPig/dgpp](https://github.com/HawkBearPig/dgpp)

DGPP started with a simple question. What happens if the serving engine, CUDA kernels, memory plan, networking, and speculative decode path are all designed around Spark instead of inherited from a general-purpose GPU stack?

The answer so far has been encouraging. On several models, DGPP is running faster than the public vLLM and SGLang results I could find for the same DGX Spark hardware and model family. It also starts quickly, has an OpenAI-compatible API, and includes deployment templates for one, two, and four Spark systems.

## Models currently supported

- **Qwen3.8-Flash-Next FP8** on two or four Sparks
- **Qwen3.8-Flash-Next NVFP4** on one or two Sparks
- **GLM-5.3-Flash FP8** on four Sparks
- **GLM-5.3-Flash NVFP4/FP8** on two or four Sparks
- **GLM-4.7 NVFP4** on four Sparks
- **Full GLM-5.3 754B int4/int8** on four Sparks
- **DeepSeek-V4.1-Flash MXFP4/FP8** on four Sparks

Here is the short version of what each quant changes. Activations stay BF16.

| Checkpoint | Dense and attention weights | Expert weights | Main exceptions |
|---|---|---|---|
| GLM-5.3-Flash FP8 | KDA and `kv_b` BF16, other DSA and dense weights FP8 | Routed and shared experts FP8 | BF16 draft, embedding, and head. BF16 cache by default, with FP8 and FP4 options |
| GLM-5.3-Flash hybrid | Same as FP8 | Routed experts in layers 3–44 NVFP4, shared experts FP8 | BF16 draft, embedding, and head. Four-Spark cache BF16, two-Spark cache FP8 |
| Qwen3.8-Flash-Next FP8 | GDN, QSA, and GR BF16 | Routed experts FP8, shared experts BF16 | FP8 n-gram table. Embedding, head, and KV cache BF16 |
| Qwen3.8-Flash-Next NVFP4 | Dense stack encoded to FP8 at load | Backbone routed experts NVFP4, MTP experts FP8, shared experts FP8 at load | BF16 embedding and KV cache. FP8 head and mapped n-gram table |
| GLM-4.7 NVFP4 | Attention BF16, dense MLP layers 0–2 NVFP4 | Routed and shared experts in layers 3–91 NVFP4 | Draft experts converted to NVFP4 at load. Embedding, head, and KV cache BF16 |
| Full GLM-5.3 int4/int8 | Layers 0–2 BF16, layers 3–77 attention int8 group-64 | Routed experts int4 group-64, shared experts int8 group-64 | Draft experts converted at load. Embedding and head BF16. BF16, FP8, or FP4 cache |
| DeepSeek-V4.1-Flash | Attention and dense weights FP8 | Routed and draft experts MXFP4, shared experts FP8 | BF16 embedding and head, FP8 Engram tables, model-native mixed cache |

## Current performance

These are warm decode results from the current production paths. The ranges come from five prompt classes, including chat, code, prose, JSON, and math. I prefer showing the spread instead of picking the fastest prompt.

| Deployment | One request | Aggregate under load | Cold prefill at roughly 2K / 8K / 32K |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next FP8, 4 Sparks | **63.2–77.7 tok/s** | **142.1–167.3 tok/s at C4** | Not re-run on the current path |
| Qwen3.8-Flash-Next FP8, 2 Sparks | **41.7–49.6 tok/s** | **69.3–83.2 tok/s at C4** | **1.299 / 5.108 / 21.168 s** |
| Qwen3.8-Flash-Next NVFP4, 2 Sparks | **62.1–74.9 tok/s** | **119.0–136.9 tok/s at C4** | **1.241 / 4.870 / 20.286 s** |
| Qwen3.8-Flash-Next NVFP4, 1 Spark | **42.6–50.3 tok/s** | **69.4–83.7 tok/s at C4** | Not re-run on the current path |
| GLM-5.3-Flash NVFP4/FP8, 4 Sparks | **50.3–58.5 tok/s** | **94.5–104.2 tok/s at C4** | **2.210 / 11.374 / 93.765 s** |
| GLM-5.3-Flash FP8, 4 Sparks | **42.2–48.7 tok/s** | **62.2–67.5 tok/s at C4** | **2.211 / 9.165 / 58.555 s** |
| GLM-4.7 NVFP4, 4 Sparks | **29.5–33.3 tok/s** | **62.7–68.7 tok/s at C4** | **2.809 / 16.865 / not measured** |
| Full GLM-5.3 754B, 4 Sparks | **25.4–29.2 tok/s** | **42.3–47.0 tok/s at C4** | **6.111 / 40.838 / 350.069 s** |
| DeepSeek-V4.1-Flash, 4 Sparks | **49.64 aggregate tok/s at C1** | **108.49 aggregate tok/s at C6** | **1,383 prompt tok/s on a 2,950-token cold prompt** |

The DeepSeek prefill result uses a different prompt workload, so it should not be compared directly with the three target-length columns used by the other models.

The full methodology, quality results, prompt lengths, and reproduction commands are in `docs/benchmarks.md` in the repository.

## How that compares with vLLM and SGLang on Spark

| Model and hardware | DGPP | Published Spark result | What stands out |
|---|---:|---:|---|
| Qwen3.8-Flash-Next NVFP4, 1 Spark | **42.6–50.3 tok/s at C1** | vLLM: **32.5 tok/s median**, 21.7 tok/s prose, 43.8 tok/s peak ([source](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476)) | Same source checkpoint. DGPP encodes the dense stack to FP8 at load. Its slowest class is close to the published peak |
| Qwen3.8-Flash-Next, 4 Sparks | FP8: **63.2–77.7 tok/s at C1**, **142.1–167.3 tok/s at C4** | vLLM NVFP4: **40.5 median**, 54.2 peak at C1, **262 tok/s at C6** in the same source above | DGPP’s full C1 range is above the published vLLM median and peak. The loaded results use different concurrency |
| Qwen3.8-Flash-Next, 2 Sparks | NVFP4: **62.1–74.9 tok/s at C1**, **119.0–136.9 tok/s at C4** | SGLang FP8: **36–41 tok/s at C1**, **88–98.5 tok/s at C4** ([source](https://forums.developer.nvidia.com/t/fp8-qwen3-8-flash-next-on-2x-dgx-spark-via-sglang-37-40-tok-s/382435)). vLLM NVFP4: **53.7 median**, 63.7 peak at C1, **309 tok/s at C6** in the Qwen source above | DGPP is above the published vLLM median across all five C1 classes and above the SGLang result at both C1 and C4. vLLM reports the higher C6 result |
| GLM-5.3-Flash, 4 Sparks | FP8: **42.2–48.7 tok/s at C1**. Hybrid: **50.3–58.5 tok/s at C1** | vLLM W4A16: **43–58 tok/s at C1** across repeated passes, with **43.0–47.3 tok/s** test means and 52.2–59.7 peaks ([source](https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350/202)) | DGPP’s hybrid sits in the upper part of the latest published vLLM band |
| GLM-4.7 NVFP4, 4 Sparks | **29.5–33.3 tok/s at C1**, **62.7–68.7 tok/s at C4** | SGLang: **24.4 tok/s at C1**, **53.2–54.6 tok/s at C4** (forum topic 366325) | DGPP is about **21–36% higher at C1** and **15–29% higher at C4** |

*I rechecked these forum results on September 16, 2026. These are separate community runs, not a controlled shootout. Prompt sets, output lengths, checkpoints, quantization, context capacity, KV format, speculative settings, concurrency, and timing scope differ. The vLLM GLM-5.3-Flash result uses a W4A16 checkpoint, a DFlash2 drafter, and a 1M context configuration. The four-Spark vLLM Qwen result uses NVFP4 while DGPP uses FP8. The two-Spark DGPP and vLLM Qwen results both use the NVIDIA NVFP4 checkpoint, but their clients and serving settings differ. The loaded comparison uses C4 for DGPP and C6 for vLLM. The SGLang row uses the larger FP8 checkpoint.*

## Where the speed comes from

DGPP does not wrap an existing Python serving backend. The scheduler, tokenizer, HTTP server, prefix cache, sampling path, collectives, model loaders, and hot inference path are native C++ and CUDA.

The CUDA work is specific to these models and to GB10. Qwen has a tiled QSA prefill kernel that shares K/V tiles across query heads. Full GLM-5.3 has a packed int4/int8 tensor-core prefill path. DeepSeek has dedicated CSA2, Engram, and DSpark execution. Dense kernels choose their lowering from the active row count instead of forcing every request shape through the same path.

Decode and speculative verification run through captured CUDA graphs. The graphs include the tensor-parallel collectives and batched request rows. DeepSeek can adjust its verification depth from the draft confidence, which avoids paying for draft rows that are unlikely to survive.

For multi-node inference, DGPP talks to libibverbs directly and stripes bulk traffic over the two active RoCE interfaces exposed by the ConnectX-7 topology. The measured aggregate link rate is about **196 Gb/s**.

Startup is another area I cared about. DGPP stores a per-rank resident weight image after the first load. Once that image exists, a deployment typically becomes ready in **15 to 30 seconds**, depending on the model. The first launch is slower because it builds the image.

## Serving and correctness

The server supports OpenAI-style Chat Completions and legacy Completions, including streaming, tool calls, JSON-schema output, reasoning content, logprobs, prefix caching, health checks, and metrics.

Performance changes go through task-level checks. Recent configurations score **38–40/40 on HumanEval**, **59–60/60 on GSM8K**, and **30/30 on schema extraction**. The larger GLM-5.3-Flash campaign uses 164 HumanEval, 300 GSM8K, and 100 extraction items.

DGPP also verifies that greedy MTP produces the same tokens as plain decode. In a multi-node run, every rank checks the ordered operation stream and must finish with the same digest.

The project is still young. The supported model list is focused, and long-context prefill still has room to improve. I keep the unsuccessful experiments and open performance gaps in the repository because they are useful when deciding what to try next.

If you have one, two, or four Sparks, I would be very interested in independent results. The repository README includes the setup steps and deployment templates. Bug reports and measurements using the same prompt set are especially welcome.
