# Token In / Token Out

!!! note

    Token In / Token Out reuses upstream vLLM `POST /inference/v1/generate`.
    vLLM Ascend does not add an Ascend-specific switch. Ordinary `vllm serve`
    exposes the endpoint; `--tokens-only` is optional and only hides the
    standard OpenAI chat/completions routes.

    Upstream references:

    - RFC: [Disaggregated Everything - Token In <> Token Out API Server](https://github.com/vllm-project/vllm/issues/22817)
    - Client example: [`token_generation_client.py`](https://github.com/vllm-project/vllm/blob/main/examples/scale_out/token_generation_client.py)

## Overview

The inference path consumes `token_ids` and returns generated token IDs.
Chat template application, tokenize, and detokenize stay on the client
unless you explicitly enable detokenize in `sampling_params`.

```text
Client tokenize
  → POST /inference/v1/generate {token_ids, sampling_params}
  → EngineCore + NPUModelRunner
  → choices[].token_ids
  → Client detokenize (optional)
```

`--tokens-only` is **not** required. The Qwen3.5-35B-A3B command below is
enough:

```bash
vllm serve /mnt/share/weights/Qwen3.5-35B-A3B \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --served-model-name auto \
  --max-num-seqs 16 \
  --port 8008
```

## Functional verification

After the server is up, run the bundled checker against port `8008` and
served name `auto`:

```bash
python tools/verify_token_in_token_out.py \
  --base-url http://127.0.0.1:8008 \
  --model auto \
  --max-tokens 16
```

The script checks:

1. `GET /v1/models` returns `auto`.
2. `POST /tokenize` produces prompt token IDs (token in).
3. `POST /inference/v1/generate` returns `choices[].token_ids` of length
   `--max-tokens` when `ignore_eos` is set (token out).
4. Streaming generate concatenates to the same length.
5. `POST /v1/completions` with a token-id `prompt` and
   `return_token_ids=true` reports matching `usage.prompt_tokens` /
   `usage.completion_tokens`.

Manual generate request:

```bash
curl http://127.0.0.1:8008/inference/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "token_ids": [151644, 8948, 198],
    "sampling_params": {
      "max_tokens": 16,
      "min_tokens": 16,
      "temperature": 0.0,
      "ignore_eos": true,
      "detokenize": false
    },
    "stream": false
  }'
```

OpenAI-compatible token I/O (same server, no `--tokens-only`):

```bash
curl http://127.0.0.1:8008/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "prompt": [151644, 8948, 198],
    "max_tokens": 16,
    "min_tokens": 16,
    "ignore_eos": true,
    "temperature": 0,
    "return_token_ids": true
  }'
```

Expected result: HTTP 200, non-empty `choices[0].token_ids`, and
`usage.prompt_tokens` equal to the input length.

## Notes

- Negative `token_ids` are rejected. Out-of-vocab IDs are undefined.
- Default generate output is token IDs, not natural language.
- Streaming chunks are incremental. Concatenate them on the client.
- Set `max_tokens` explicitly. Do not rely on server defaults.
- Model Runner V1 and V2 share the same HTTP contract.

## Related features

- [vLLM-Ascend for RL](rl.md)
