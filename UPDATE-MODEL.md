# Adding a New Model

This document covers what to change when a new model appears in the OpenHands index,
and what additional work is needed when the model introduces a new architecture.

The pipeline can detect unmapped open-weights models via `check_missing_mappings()`
in `main.py`. Note: this check relies on the `openness` field in benchmarks.json,
which was added after earlier snapshots were generated — it may miss models in
snapshots that pre-date the field.

---

## Checklist for a New Open-Weights Model

### 1. Display name mapping (if the index name differs)

**File:** `pipeline/pipeline/snapshots/reader.py` — `METADATA_NAME_OVERRIDES`

The `"model"` field in `metadata.json` may not match the canonical name we want
to show users (e.g. `"Minimax-2.7"` submitted under the `MiniMax-M2.7`
directory). Add an entry mapping the raw metadata value to the canonical name:

```python
METADATA_NAME_OVERRIDES: dict[str, str] = {
    ...
    "Minimax-2.7": "MiniMax-M2.7",
}
```

Omit this step when the `"model"` field already matches the canonical name.

**`alias_map.py` is different:** `_RENAMES` in `alias_map.py` is only for
models whose results *directory* was renamed in the upstream git history (e.g.
`jade-spark-2862` → `Minimax-2.5` → `MiniMax-M2.5`). Each entry there should
correspond to an actual git commit that renamed the directory. Do not use it to
correct a metadata field mismatch.

### 2. HuggingFace mapping

**File:** `pipeline/pipeline/sources/huggingface.py` — `MODEL_NAME_TO_HF_ID`

Add the model's canonical name and its HuggingFace repo ID:

```python
MODEL_NAME_TO_HF_ID: dict[str, str] = {
    ...
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
}
```

Only open-source/open-weight models belong here. Closed-source models (GPT,
Claude, Gemini, etc.) are never added.

The pipeline fetches `config.json` from the HF repo to derive architecture
parameters.

**If the model has no HuggingFace page** (or its page lacks `config.json`),
add it to `MODEL_ARCH_SOURCE_HF_ID` instead, pointing to a compatible model
whose config to use for calculations:

```python
MODEL_ARCH_SOURCE_HF_ID: dict[str, str] = {
    "MiniMax-M2.7": "MiniMaxAI/MiniMax-M2.5",  # borrow M2.5's config
}
```

And provide a link URL for users (GitHub release, blog post, etc.)
via `MODEL_ALT_URL`:

```python
MODEL_ALT_URL: dict[str, str] = {
    "MiniMax-M2.7": "https://github.com/MiniMax-AI/MiniMax-M2.7",
}
```

When `MODEL_ARCH_SOURCE_HF_ID` is set for a model, `MODEL_NAME_TO_HF_ID` is not
required — `fetch_all_models()` will process it automatically. The output
`ModelSpec` will have `hf_model_id=None` and `model_url` set to the alt URL,
which the frontend uses as the link target instead of the HF page.

### 3. License info

**File:** `pipeline/pipeline/sources/huggingface.py` — `MODEL_LICENSE_INFO`

Add the license name and a direct URL to the license text. The key is the
canonical model name (same as used in `MODEL_NAME_TO_HF_ID`):

```python
MODEL_LICENSE_INFO: dict[str, tuple[str, str]] = {
    ...
    "MiniMax-M2.7": (
        "Non-commercial License",
        "https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE",
    ),
}
```

The pipeline will raise a `ValueError` at runtime if this entry is missing for
any model in `MODEL_NAME_TO_HF_ID`. The CI test `test_all_models_have_license_info`
in `tests/test_license.py` also enforces this.

---

## Checklist for a New Architecture

When a new model introduces a `model_type` not yet known to the pipeline,
`fetch_model()` will raise `UnsupportedArchitecture`. Work through the steps below.

### 1. Identify the model_type

Fetch the model's `config.json` from HuggingFace and check the `"model_type"` field.
This is the key used in `KNOWN_ARCHITECTURES`.

### 2. Register the architecture

**File:** `pipeline/pipeline/sources/param_counter.py` — `KNOWN_ARCHITECTURES`

Add an entry mapping the `model_type` string to its attention type and MoE field
mapping. The two attention types are:

- `AttentionType.GQA` — standard grouped-query attention (most models)
- `AttentionType.MLA` — multi-head latent attention (DeepSeek-style)

For a standard GQA MoE model:

```python
KNOWN_ARCHITECTURES: dict[str, tuple[AttentionType, MoEFieldMapping]] = {
    ...
    "new_model_type": (
        AttentionType.GQA,
        MoEFieldMapping(
            expert_count_key="num_experts",           # key for routed expert count
            shared_expert_key="shared_expert_intermediate_size",
            shared_expert_is_count=False,             # True if key is a count, False if it's a size
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key=None,                    # key for number of dense (non-MoE) layers, or None
            mtp_key=None,                             # key for MTP/speculative decoding layers, or None
        ),
    ),
}
```

Look at the model's `config.json` to find the correct key names. Compare against
existing entries for similar architectures (e.g. `qwen3_moe`, `deepseek_v32`).

For a Dense model (no MoE), pass an empty `MoEFieldMapping()` — the counter
will still work correctly because it falls back to the dense path.

### 3. Add a test config

**File:** `pipeline/tests/test_param_counter.py`

Add a trimmed copy of the model's `config.json` as a module-level constant
(following the pattern of `DEEPSEEK_V32_CONFIG`, `GLM_47_CONFIG`, etc.) and
write tests for total params, active params, and model type. This is the single
source of truth for test configs — other test files import from here.

---

## Verification

After making changes, run the full test suite:

```
cd pipeline && python -m pytest tests/ -v
```

The key tests that enforce consistency:

| Test | What it checks |
|------|---------------|
| `test_all_models_have_license_info` | Every model in `MODEL_NAME_TO_HF_ID` has a license entry |
| `test_modelspec_fields_match_typescript` | `ModelSpec` fields stay in sync with the TypeScript `Model` interface |
| `test_param_counter.py` | Architecture-specific param counts are correct |
| `test_hf_kv_fields.py` | KV cache fields are extracted correctly for each attention type |
