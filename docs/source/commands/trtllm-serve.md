# Deploy model with TRT-LLM OpenAI-Compatible Server

TRT-LLM provides the openai-compatiable API via `trtllm-serve` command.
 A complete reference for the API is available in the OpenAI API Reference.


 This tutorial covers below topics
 * Run trtllm-serve with NGC containers
 * Launch the OpenAI-Compatibale Server
 * Usage of `extra_llm_api_options` knob
 * Run the Performance benchmark.


1. Launch the NGC container

TRT-LLM deploys the pre-built container on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags).
There're t
```bash
docker run --rm --ipc host -p 6666 --gpus all -it nvcr.io/nvidia/tensorrt-llm/release:0.21.0rc1
```


2. Deploy the OpenAI-Compatiable Server

`trtllm-serve` is a built-in tool which provides the OpenAI-Compatible APIs for launching the service with self-hosted models.
Below section lists how to deploy model on single node or multi-nodes with `trtllm-serve`.
   * Single Node

Launch the server with TinyLlama-1.1B-Chat-v1.0 on ip address 10.63.131.230:6666.
```bash
trtllm-serve TinyLlama-1.1B-Chat-v1.0 --port 6666  --host 10.63.131.230 --backend pytorch
```
   * Multi-Node with Slurm
```bash
echo -e "enable_attention_dp: true\npytorch_backend_config:\n  enable_overlap_scheduler: true" > extra-llm-api-config.yml

srun -N 2 -w [NODES] \
    --output=benchmark_2node.log \
    --ntasks 16 --ntasks-per-node=8 \
    --mpi=pmix --gres=gpu:8 \
    --container-image=<CONTAINER_IMG> \
    --container-mounts=/workspace:/workspace \
    --container-workdir /workspace \
    bash -c "trtllm-llmapi-launch trtllm-serve deepseek-ai/DeepSeek-V3 --backend pytorch --max_batch_size 161 --max_num_tokens 1160 --tp_size 16 --ep_size 4 --kv_cache_free_gpu_memory_fraction 0.95 --extra_llm_api_options ./extra-llm-api-config.yml"
```

3. trtllm-serve Arguments
   trtllm-serve leverage `extral_llm_api_options` knob to overwtie the parameters specified by trtllm-serve.
   Generally, we create a yaml file which consists of knobs.
   e.g
   ```yaml
     use_cuda_graph: true
     cuda_graph_padding_enabled: true
     print_iter_log: true
     kv_cache_dtype: fp8
     enable_attention_dp: true
   ```

   3.1 Performance knobs list
   * use_cuda_graph
   * cuda_graph_padding_enabled, use CUDA graphs for decoding. Default is False.
   * cuda_graph_batch_sizes: List of batch sizes to create CUDA graphs for. Default Value is None.
   * print_iter_log: Print iteration logs. Default value is False.
   * .... [TODO-guomingz] enum all knobs here
   * autotuner_enabled: Enable autotuner only when torch compile is enabled.Default Value is True.

   3.2 Hot Model's config.

   * [TODO-guomingz] Add DS/LLAMA models config for SA

4. Performance Benchmarking (TODO: add detailed configuration here for sa reproducing results.)
For OpenAI-Compatibe Serve, we recommend use [genai-perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) to test performance.

Install the genai-perf toolset firstly.
```bash
pip install genai-perf
```

Use below script to run benchmark with [DeepSeek-R1-0528-FP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4)
```bash
#! /usr/bin/env bash

genai-perf profile \
    -m DeepSeek-R1-0528-FP4 \
    --tokenizer nvidia/DeepSeek-R1-0528-FP4 \
    --endpoint-type chat \
    --random-seed 123 \
    --synthetic-input-tokens-mean 128 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --request-count 100 \
    --request-rate 10 \
    --profile-export-file my_profile_export.json \
    --url localhost:8000 \
    --streaming
```
