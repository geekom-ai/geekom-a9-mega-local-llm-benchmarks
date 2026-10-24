> **⚠️ Mirror — archived snapshot.** The canonical, actively maintained repository is **[GEEKOMOfficial/geekom-a9-mega-local-llm-benchmarks](https://github.com/GEEKOMOfficial/geekom-a9-mega-local-llm-benchmarks)**. Please cite numbers from, and file issues in, the canonical repository. This copy is kept read-only for reference and will not be updated.

# Local LLM Inference on GEEKOM A9 Mega

This repository answers three questions for end customers, using data measured on physical GEEKOM hardware:

1. When running a large language model locally on a mini PC, **which quantization level should you choose?**
2. Once that level is chosen, **what performance can each GEEKOM A9 Mega model — the A9 Mega 395 (128 GB) and the A9 Mega 388 (64 GB) — actually deliver?**
3. **Which model should you pick for each device?**

The short answer: **4-bit quantization is the sweet spot between file size and model fidelity**, and it is the level used for the recommended configurations in this repository.

Section 5 turns those results into a concrete shortlist: the **strongest 4-bit recommendation for the A9 Mega 395** and **three recommended models for the A9 Mega 388**. The 395 is the higher-spec model, with more CPU cores and twice the memory (128 GB vs. 64 GB); the 388 is the more cost-effective choice. The recommended models have also been run on the 395, and the 395 can host larger models — including the 111.4 GB flagship — on a single device, where the 388 needs a second A9 Mega for it. The shortlist highlights useful choices rather than covering every supported model.

All performance data was measured on physical GEEKOM units. **Device configurations, model filenames, and recorded launch commands are published with the results**, and anything not tested is marked as such rather than assumed.

Full records for each device, including test commands, VRAM usage, and notes:

- A9 Mega 395 — all 37 test records: [`docs/A9-Mega-395/`](./docs/A9-Mega-395/)
- A9 Mega 388 — all 35 test records: [`docs/A9-Mega-388/`](./docs/A9-Mega-388/)

This root README summarizes the benchmarks. Device-specific installation guides can be added to those folders when the setup steps have been verified.

## Model performance at a glance

Headline short-context (128-token input) results for the models tested on the A9 Mega 395 and A9 Mega 388, for a quick overview before the detailed sections. See [Section 4](#4-measured-4-bit-performance-by-device) for longer context lengths, deployment, and per-device detail.

| Model / Quantization | Model Size (GB) | Generation Speed (token/s) | Prompt Processing (token/s) | TTFT (s) |
| :---: | :---: | :---: | :---: | :---: |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | 111.4 | 32.43 | 132.22 | 0.97 |
| DeepSeek-V4-Flash-0731 `UD-Q2_K_XL` | 96.8 | 14.71 | 129.24 | 0.99 |
| Qwen3.6-35B-A3B `UD-Q4_K_M` | 22.7 | 68.03 | 822.22 | 0.16 |
| Gemma-4-31B `q4_0` | 16.4 | 11.28 | 26.83 | 4.77 |
| Qwen3.8-27B `FP4-FAST` | 14.6 | 31.29 | 318.64 | 0.40 |
| MiniCPM5-2B `Q4_K_M` | 1.56 | 103.10 | 1320.23 | 0.10 |

Source: recorded test data in the device archives ([A9 Mega 395](./docs/A9-Mega-395/), [A9 Mega 388](./docs/A9-Mega-388/)). Figures are for single-device runs; the larger models were also measured in multi-node cascades, and decode rate falls as context grows.

## Contents

1. [Quantization level vs. memory requirement](#1-quantization-level-vs-memory-requirement)
2. [Quantization level vs. model fidelity](#2-quantization-level-vs-model-fidelity)
3. [Conclusion: 4-bit is the sweet spot](#3-conclusion-4-bit-is-the-sweet-spot)
4. [Measured 4-bit performance by device](#4-measured-4-bit-performance-by-device)
5. [Recommended models by device](#5-recommended-models-by-device)

---

## 1. Quantization level vs. memory requirement

Quantization means **representing model weights with fewer bits** (often summarized as *bits per weight*, or **bpw**). Labels such as `Q4_K_XL` name quantization schemes; they do not mean every weight uses exactly four bits. For **Qwen3.8-Flash-Next**, [Unsloth's hardware requirements table](https://unsloth.ai/docs/models/qwen3.8-next) gives these approximate total memory requirements (RAM plus VRAM, or unified memory):

| Quantization level | 1-bit | 2-bit | 3-bit | 4-bit | 5-bit | 8-bit | BF16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Typical GGUF quantization | `IQ1_S` / `IQ1_M` | `Q2_K_XL` | `IQ3_XXS` / `Q3_K_XL` | `IQ4_XS` / `Q4_K_XL` | `Q5_K_XL` | `Q8_0` | BF16 |
| Approximate total memory requirement | 75 GB | 79 GB | 90 GB | 96–114 GB | 163 GB | 200 GB | 355 GB |

Three points follow from this table:

- **Required memory grows with quantization level**: the more precision used per weight, the more storage and runtime memory the model needs. Exact requirements depend on the quantization scheme and runtime settings.
- **Memory grows gently at low quantization levels**: 1-bit to 4-bit only moves from 75 GB to 96–114 GB (roughly +30% to +50%), yet that is already enough to hold this hundred-gigabyte-class flagship model.
- **Memory expands sharply at high quantization levels**: BF16 (355 GB) is more than three times the 4-bit range, and 8-bit (200 GB) is likewise far beyond the memory capacity of a mainstream mini PC. Both place an order-of-magnitude higher demand on memory capacity and memory bandwidth.

**What this means when choosing a device**: the model weights, draft model, runtime overhead, and context cache (KV cache) all need memory. The recorded GEEKOM setup uses a 2.8 GB draft model for MTP. The 111.4 GB 4-bit main model was recorded on a single A9 Mega 395 through 128K input tokens (Section 4.2); part of its memory use is offloaded to CPU (system) RAM (the model's N-gram embedding data), and because that portion can use system virtual memory (swap pages), the combined footprint can exceed the listed 128 GB memory capacity. Memory capacity affects both the model size and context length a device can serve.

## 2. Quantization level vs. model fidelity

[Unsloth's Qwen3.8-Flash-Next quantization analysis](https://unsloth.ai/docs/models/qwen3.8-next#quantization-analysis) reports the following GGUF file sizes and fidelity metrics. Its chart titles abbreviate the model name to “Qwen3.8-Flash”; the data appears within the Flash-Next guide. **Top-1** measures recovery of the reference model's first-choice output token, and **mean KLD** measures the difference between output distributions (lower is better). These are model-fidelity metrics, not task accuracy scores.

![Unsloth Qwen3.8-Flash-Next GGUF quantization data: Top-1 recovery rises and mean KLD falls as file size increases. The 111.3 GB UD-Q4_K_XL point is highlighted.](docs/images/qwen3.8-flash-next-quantization-fidelity.svg)

| GGUF quantization | Disk size (GB) | Top-1 recovery (%) | Mean KLD |
| --- | ---: | ---: | ---: |
| `UD-IQ1_S` | 72.5 | 77.325 | 0.396070 |
| `UD-IQ1_M` | 74.5 | 79.691 | 0.314739 |
| `UD-Q2_K_XL` | 78.9 | 82.715 | 0.224607 |
| `UD-IQ3_XXS` | 82.0 | 85.414 | 0.165120 |
| `UD-Q3_K_XL` | 90.0 | 88.315 | 0.106504 |
| `UD-IQ4_XS` | 93.7 | 89.554 | 0.083630 |
| `UD-Q4_K_XL` | 111.3 | 92.255 | 0.046893 |
| `UD-Q5_K_XL` | 158.3 | 93.680 | 0.030415 |
| `UD-Q6_K_XL` | 169.2 | 94.089 | 0.027091 |
| `Q8_0` | 188.2 | 94.122 | 0.026574 |

These are Unsloth's published figures for its quantizations. They do not measure accuracy on user tasks or compare different model families.

## 3. Conclusion: 4-bit is the sweet spot

The Unsloth memory requirements in Section 1 and its GGUF fidelity measurements in Section 2 support these observations for Qwen3.8-Flash-Next:

1. **Fidelity and size are positively correlated**: higher quantization gives higher Top-1 recovery and lower KLD, and the two metrics corroborate each other.
2. **Marginal returns diminish above ~111 GB (`UD-Q4_K_XL`)**: `UD-Q5_K_XL` (158.3 GB, 93.680% Top-1, 0.030415 KLD) and `Q8_0` (188.2 GB, 94.122% Top-1, 0.026574 KLD) add only about 1%–2% more Top-1 and a KLD improvement from about 0.05 to about 0.03 — for 1.4×–1.7× the 4-bit size, a cost out of proportion to the gain.
3. **Low-bit quantization costs a lot**: `UD-IQ1_S` (72.5 GB) reaches only 77.325% Top-1 with 0.396070 KLD, and `UD-Q2_K_XL` (78.9 GB) reaches 82.715% Top-1 with 0.224607 KLD — capacity saved, fidelity lost.
4. **At the same file size, the quantization scheme matters**: `UD-IQ4_XS` is 93.7 GB with 89.554% Top-1, while `UD-Q4_K_XL` is 111.3 GB with 92.255% Top-1. The two “4-bit” choices differ in size and fidelity.

> **4-bit quantization such as `UD-Q4_K_XL` is therefore the sweet spot**: it delivers near-saturated fidelity (Top-1 recovery about 92%, KLD below 0.05) at 111.3 GB, close to the 111.4 GB model size recorded in the A9 Mega tests. Unsloth's fidelity figures should not be assumed for other models or quantizations.
>
> This is why the main device recommendations in this repository **use 4-bit quantization**. This does not establish the best quantization for every model or task. Non-4-bit models are documented separately; the 2-bit `DeepSeek-V4-Flash-0731-UD-Q2_K_XL` is also discussed as an alternative for the 395 in Section 5.1.

## 4. Measured 4-bit performance by device

### 4.1 Devices and hardware configuration

| Item | A9 Mega 395 | A9 Mega 388 |
| --- | --- | --- |
| Processor | AMD Ryzen AI MAX+ 395 | AMD Ryzen AI MAX+ 388 |
| CPU cores / threads | ![16 cores / 32 threads](docs/images/a9-mega-395-cpu.svg) | ![8 cores / 16 threads](docs/images/a9-mega-388-cpu.svg) |
| Graphics | Radeon 8060S | Radeon 8060S |
| Memory bandwidth | 256 GB/s | 256 GB/s |
| Memory size | ![128 GB](docs/images/a9-mega-395-memory.svg) | ![64 GB](docs/images/a9-mega-388-memory.svg) |
| Operating system | Ubuntu 26.04 | Ubuntu 26.04 |
| Graphics driver | 31.50.0.0 | 31.50.0.0 |


> Both devices use the same listed graphics processor and memory bandwidth. The **395 has twice the CPU cores and threads (16/32 vs. 8/16) and twice the memory (128 GB vs. 64 GB)**, making it the higher-spec model; the **388 is the more cost-effective option**. Memory capacity determines the model sizes a single device can hold and the context lengths it can serve. In the recorded tests the 395's additional memory allowed the featured 111.4 GB model to run on one device, while the 388 used a two-device cascade for that model.

### 4.2 Primary model on A9 Mega 395: Qwen3.8-Flash-Next UD-Q4_K_XL

A9 Mega 395 single-device test conditions:

| Item | Configuration |
| --- | --- |
| Model | `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf` (111.4 GB, the upper end of the 4-bit range in Section 1) + draft model `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (2.8 GB) |
| Acceleration | `Yes(MTP)`, `--spec-type draft-mtp --spec-draft-n-max 2` |
| Context length | `-c 256000`, `-fa on`; measured through 128K input tokens |
| Batch settings | `-b 4096 -ub 2048 -np 1` |
| Deployment | Single device (`llama-server`, without `--rpc`) |

![Line plot of A9 Mega 395 single-device decode rate: 32.43 tokens/s at 128 input tokens, 28.48 at 8K, 25.88 at 16K, 23.58 at 32K, 18.06 at 64K, and 15.45 at 128K.](docs/images/a9-mega-395-decode-rate.svg)

The points are spaced evenly by measured setting for readability; their horizontal distances are not proportional to token count. The [A9 Mega 395 test archive](./docs/A9-Mega-395/) has the corresponding TTFT, prefill, output length, and VRAM measurements.

The single device decoded at **32.43 tokens/s at 128 input tokens**, **23.58 tokens/s at 32K**, and **15.45 tokens/s at 128K**. Time to first token rose from 0.97 s at 128 input tokens to 720.61 s at 128K.

The reported memory usage for this single-device run is given as GPU VRAM plus a portion offloaded to CPU (system) RAM (for example, `101.61 (+30.17)` GB, where the offloaded part holds the model's N-gram embedding data). Because the offloaded portion can use system virtual memory (swap pages), the combined 131.78–135.06 GB can exceed the listed 128 GB memory size; the archive retains those readings as measured.

### 4.3 A9 Mega 388: Qwen3.6-35B-A3B and other 4-bit models

The recommended single-device `Qwen3.6-35B-A3B-UD-Q4_K_M` (22.7 GB, MTP N=2) was measured at seven input lengths:

![Line plot of A9 Mega 388 single-device decode rate: 68.03 tokens/s at 128 input tokens, 67.76 at 8K, 63.55 at 16K, 54.30 at 32K, 50.30 at 64K, 32.61 at 128K, and 21.66 at 256K.](docs/images/a9-mega-388-decode-rate.svg)

The points are spaced evenly by measured setting for readability; their horizontal distances are not proportional to token count. At 256K, the model decoded at **21.66 tokens/s** with a **645.04 s** time to first token. The [A9 Mega 388 test archive](./docs/A9-Mega-388/) has the full measurements and single-device command.

The table below summarizes three 4-bit models through 128K input tokens:

| Model | Size | Acceleration | 128 | 8K | 16K | 32K | 64K | 128K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-UD-Q4_K_XL | 17.6 GB | MTP N=4 | 20.24 | 21.30 | 19.53 | 16.87 | 12.86 | 9.88 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 22.7 GB | MTP N=2 | 68.03 | 67.76 | 63.55 | 54.30 | 50.30 | 32.61 |
| MiniCPM5-2B-Q4_K_M | 1.56 GB | No | 103.10 | 87.08 | 75.15 | 61.74 | 44.76 | 29.03 |

Corresponding time to first token (s):

| Model | 128 | 8K | 16K | 32K | 64K | 128K |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-UD-Q4_K_XL | 0.44 | 24.54 | 52.44 | 109.55 | 268.94 | 646.85 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 0.16 | 6.61 | 14.79 | 32.51 | 103.58 | 218.14 |
| MiniCPM5-2B-Q4_K_M | 0.10 | 3.09 | 7.52 | 21.14 | 70.63 | 255.42 |

**Reading the results**:

- **MoE models around 35B-A3B are the 388's comfort zone**: `Qwen3.6-35B-A3B-UD-Q4_K_M` (22.7 GB) decodes at 54.30 tokens/s at 32K, 32.61 tokens/s at 128K, and 21.66 tokens/s at 256K.
- 27B-class dense models drop to 5–10 tokens/s at long context, so they suit medium context lengths.
- The tested 2B model (`MiniCPM5-2B-Q4_K_M`, a 1.56 GB file with reported VRAM usage of roughly 15–18 GB) reaches 103.10 tokens/s at 128 input tokens; its decode rate falls to 29.03 tokens/s at 128K.

> **Scope note**: the [388 archive](./docs/A9-Mega-388/) records every tested configuration. Only the three 388 models listed in **Section 5.2** are promoted to the recommendation shortlist; the other configurations, including `Qwen3.8-27B-UD-Q4_K_XL` and the triple-node cascade, remain measured records.

### 4.4 Other 4-bit models on the A9 Mega 395 (decode rate, tokens/s)

| Model | Acceleration | 128 | 8K | 16K | 32K | 64K | 128K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-FP4-FAST | MTP N=4 | 31.29 | 25.43 | 24.51 | 20.28 | 15.42 | 8.95 |

Corresponding time to first token (s):

| Model | 128 | 8K | 16K | 32K | 64K | 128K |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-FP4-FAST | 0.40 | 23.56 | 51.11 | 118.52 | 306.21 | 865.62 |

> **Scope note**: the 395 was also tested with the **2-bit** `DeepSeek-V4-Flash-0731-UD-Q2_K_XL` (96.8 GB, in single-device, dual-node cascade, and triple-node cascade deployments), which is not 4-bit, so it is not listed in this section. Within the 4-bit results, `Qwen3.8-Flash-Next-UD-Q4_K_XL` remains the primary recommendation in **Section 5.1**, while `Qwen3.8-27B-FP4-FAST` is kept as a measured record. Full records are in [`docs/A9-Mega-395/`](./docs/A9-Mega-395/).

### 4.5 Reproduction commands

<details>
<summary>395 — single device, Qwen3.8-Flash-Next UD-Q4_K_XL</summary>

```bash
./llama-server -m /home/user/mode/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 -c 256000 -b 4096 -ub 2048 -fa on -np 1 --host 127.0.0.1 --port 8080 --metrics -lv 4 -md /home/user/mode/Qwen3.8-Flash-Next-UD-Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf -ngld 99 --spec-type draft-mtp --spec-draft-n-max 2
```

</details>

<details>
<summary>388 — dual-node cascade, Qwen3.8-Flash-Next UD-Q4_K_XL</summary>

```bash
# On the worker
LD_LIBRARY_PATH="$PWD:$PWD/lib:$PWD/bin:$LD_LIBRARY_PATH" && ./ggml-rpc-server -H 192.168.10.10 -p 50052 -t 16

# On the host (after starting the worker)
./llama-server -m /home/user/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 -c 256000 -b 2048 -ub 1024 -fa on -np 1 --rpc 192.168.10.10:50052 --host 127.0.0.1 --port 8080 --metrics -lv 4 -md /home/user/Qwen3.8-Flash-Next-UD-Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf -ngld 99 --spec-type draft-mtp --spec-draft-n-max 2
```

</details>

The complete test command for every model is in the corresponding device archive.

These are recorded test commands with machine-specific file paths and network addresses. Reproduction requires the corresponding model files, a compatible llama.cpp build, and paths and addresses appropriate to the test environment.

---

## 5. Recommended models by device

The 4-bit results in Section 4 reduce to a short, device-specific shortlist. The 395's larger memory allows the featured model to run on one device, while the 388 requires a second A9 Mega for it. The two smaller models recommended for the 388 were also tested on the 395, but those results are not included in this repository, so no 395 performance figures are quoted for them here. These recommendations do not exhaust either device's possible configurations.

### 5.1 A9 Mega 395 — primary recommendation and another capable option

| Role | Model | File size | Deployment |
| --- | --- | --- | --- |
| Strongest model | `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf` | 111.4 GB | Single device |

The 395 was recorded running this hundred-gigabyte-class 4-bit flagship **on a single device** — no cascade required (Section 4.2). It decoded at 32.43 tokens/s at 128 input tokens and 15.45 tokens/s at 128K. Its reported memory usage (GPU VRAM plus a portion offloaded to CPU RAM, which can use system virtual memory) can exceed the listed 128 GB memory capacity.

**DeepSeek-V4-Flash-0731 is another competitive model choice** for the 395. Its `UD-Q2_K_XL` version (96.8 GB, 2-bit) was tested on a single device through 256K input tokens; it decoded at 14.71 tokens/s at 128 tokens, 9.71 tokens/s at 128K, and 7.16 tokens/s at 256K ([395 test archive](./docs/A9-Mega-395/)). It sits outside the 4-bit recommendation table, and this repository does not provide a head-to-head intelligence benchmark between it and Qwen3.8-Flash-Next.

### 5.2 A9 Mega 388 — three recommended models

| # | Role | Model | File size | Deployment | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | Flagship (needs a second device) | `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf` | 111.4 GB | Dual-node cascade with a second A9 Mega | The 64 GB 388 cannot hold this model alone; in a two-node cascade it decoded at 30.76 tokens/s at 128 tokens (see the [388 test archive](./docs/A9-Mega-388/)). |
| 2 | Balanced everyday default | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22.7 GB | Single device | MoE 35B-A3B; 54.30 tokens/s at 32K, 32.61 at 128K, and 21.66 at 256K (Section 4.3). |
| 3 | Lightweight, high-throughput | `MiniCPM5-2B-Q4_K_M.gguf` | 1.56 GB | Single device | High throughput at short context: 103.10 tokens/s at 128 input tokens, falling to 29.03 tokens/s at 128K (Section 4.3). |

> **Compatibility note**: the flagship has published test records for both devices. The two smaller models were also tested on the 395, but those records have not been added to this repository; their 395 performance is not reported here.

**In short**: pick the higher-spec **A9 Mega 395** (128 GB) for larger single-device models, including the 111.4 GB Qwen flagship or the tested DeepSeek alternative. Pick the more cost-effective **A9 Mega 388** (64 GB) for an everyday high-throughput assistant — its 35B-A3B model still decodes at 54.30 tokens/s at 32K — and add a second A9 Mega to reach the same flagship through a dual-node cascade.

> These conclusions apply only to the configurations recorded in this repository. Changing software versions, models, context length, or power profile may change the results, and data from a single device should not be generalized into guaranteed performance across all GEEKOM configurations.

## Known limitations

The table separates observed limits from details that still need verification.

| Item | Limitation or open item |
| --- | --- |
| Maximum tested context | The featured **A9 Mega 395 single-device** run was measured through 128K input tokens (`-c 256000` configured); a 256K single-device result is not recorded. The **A9 Mega 388** single-device `Qwen3.6-35B-A3B-UD-Q4_K_M` run was measured through 256K. Other models and deployments have different measured limits; see the device archives. |
| Memory capacity | The **388 (64 GB) cannot hold the 111.4 GB 4-bit flagship model on a single device**, so the 388 requires a **dual-node cascade**. The **395 (128 GB)** ran it on a single device as measured. Single-device models are otherwise limited to the physical memory (64 GB on the 388, 128 GB on the 395); the 395's flagship run exceeded 128 GB only by offloading part of the model to CPU RAM, which can use system virtual memory (swap pages). |
| Long-context throughput | Decode rate falls and time to first token rises sharply as context grows on both devices. |
| Test conditions not yet published | BIOS, NPU driver, and power profile are still "To be confirmed". |
| Offline use | Local operation does not mean every step works offline: models, software, dependencies, and required resources must be downloaded and configured before the device is disconnected from the internet. |

## Responsible use

This is an experimental reference project, not a guarantee that every model or third-party application will work on every configuration. Model outputs may be inaccurate. Do not use generated output as the sole basis for medical, legal, financial, safety-critical, or other high-impact decisions.

The performance data here was measured on specific, disclosed configurations. Background applications, software versions, model settings, ambient conditions, and power profiles can affect performance, so results from one device should not be read as guaranteed performance for every GEEKOM configuration.

Support scope and issue-response expectations will be added before public release.

## Related resources

- [GEEKOM Resource Center](https://github.com/GEEKOMOfficial/geekom-resource-center)
- [GEEKOM Official Website](https://www.geekompc.com/)
- [GEEKOM Support](https://www.geekompc.com/support/)
