# Static Expert Placement in llama.cpp

## Scope and question

Question: what would it take to add static expert-level placement to llama.cpp,
using SGLang's expert-parallel structure as a reference, for heterogeneous GPUs
connected over PCIe x1?

This note deliberately excludes tensor parallelism, GPU cache migration, dynamic
expert migration, replication, and optimized collectives. The initial target is
stationary expert weights and the minimum activation traffic needed to execute a
routed expert on its owner GPU.

## Executive conclusion

llama.cpp already has the right logical routing representation, but its execution
and placement units are too coarse:

```text
Current:
  MoE layer -> one or more [input, output, expert] projection tensors
            -> two routed GEMMs (merged gate/up + down), or three separate GEMMs
            -> one backend/device for each node and its complete weight tensor

Proposed:
  router -> expert_to_owner[global_expert]
         -> expert_to_local[global_expert]
         -> group selected activation rows by owner GPU
         -> copy only those activation rows to owner GPU
         -> execute owner-local expert GEMMs using stationary packed weights
         -> combine selected local experts per owner GPU
         -> copy one owner partial result back
         -> combine owner partial results on the origin GPU
```

The smallest robust prototype is therefore a host-orchestrated static MoE
executor for one layer, rather than a first attempt to teach the generic graph
scheduler that one tensor has multiple device owners. Once correctness and
traffic are measured, the same grouping can be made into a graph-native/custom
operation.

## llama.cpp call path

### 1. Expert representation and loading

The model metadata records `n_expert` and `n_expert_used` in
[`src/llama-model.cpp:1116`](llama.cpp/src/llama-model.cpp) and validates them in
the same file. Expert tensors are created by
`llama_model_base::create_tensor_gate_up_exps()` at
[`src/llama-model.cpp:2931`](llama.cpp/src/llama-model.cpp). The merged form is
`[n_embd, 2*n_ff, n_expert]`; separate gate/up/down forms also carry experts in
dimension 2. Names are `blk.%d.ffn_{gate,up,down}_exps` from
[`src/llama-arch.cpp:422`](llama.cpp/src/llama-arch.cpp), with tensor kinds
declared in [`src/llama-arch.h:448`](llama.cpp/src/llama-arch.h).

The generic loader associates these tensors with the repeating layer's buffer
list through `llama_model_base::create_tensor()` at
[`src/llama-model.cpp:1708`](llama.cpp/src/llama-model.cpp). Per-expert scales
are separate tensors shaped `[n_expert]` and are created around
[`src/llama-model.cpp:1448`](llama.cpp/src/llama-model.cpp).

### 2. Layer/device placement

`llama_model_base::load_tensors()` computes layer split points and
`get_layer_buft_list()` at [`src/llama-model.cpp:1359`](llama.cpp/src/llama-model.cpp)
assigns each complete layer to one device under layer split mode. `dev_layer[il]`
is one `{device, buft_list}` pair per layer. This is the current assumption that
all weights needed by an MoE operation in a layer share a placement. There is no
`expert_to_device[]` field in `llama_layer`, `llama_model::impl`, or the GGUF
metadata path.

llama.cpp also has `LLAMA_SPLIT_MODE_ROW`, which can put tensor-parallel split
buffers into the layer buffer list (`make_gpu_buft_list()` at
[`src/llama-model.cpp:977`](llama.cpp/src/llama-model.cpp)). That mode is
intentionally out of scope here. The prototype must use layer split mode and
must not rely on row-split behavior.

Weights are marked `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` around
[`src/llama-model.cpp:1656`](llama.cpp/src/llama-model.cpp), which lets the
scheduler prefer the backend containing a weight. That is a whole-buffer
property, not an expert-slice ownership mechanism.

### 3. Router and routed expert GEMMs

The MoE graph is built in `llm_build_moe_ffn()` in
[`src/llama-graph.cpp:1928`](llama.cpp/src/llama-graph.cpp). It computes router
logits, probabilities, optional group masking, and top-k IDs. The selected IDs
have shape `[n_expert_used, n_tokens]`; routing weights have the matching
selected-expert/token shape.

The normal graph reshapes the input to `[n_embd, 1, n_tokens]` and relies on the
broadcast rule of `MUL_MAT_ID`; it calls
`build_lora_mm_id()` for gate/up and down at approximately
[`src/llama-graph.cpp:2095`](llama.cpp/src/llama-graph.cpp) and
[`src/llama-graph.cpp:2231`](llama.cpp/src/llama-graph.cpp). Those become
`GGML_OP_MUL_MAT_ID` nodes. The operation's contract is documented in
[`ggml/src/ggml.c:3318`](llama.cpp/ggml/src/ggml.c):

```text
as  = [cols, rows, n_expert]
b   = [cols, n_expert_used, n_tokens]
ids = [n_expert_used, n_tokens]
c   = [rows, n_expert_used, n_tokens]
```

`ggml_mul_mat_id()` indexes expert `as[:,:,ids[e,t]]`, but has no device owner
argument. The down result is weighted and reduced by views/adds in
[`src/llama-graph.cpp:2238`](llama.cpp/src/llama-graph.cpp).

Llama 4 is an important exception: it explicitly repeats the input and applies
routing weights before expert execution (`weight_before_ffn` at
[`src/llama-graph.cpp:1944`](llama.cpp/src/llama-graph.cpp)). Other supported
architectures can also introduce per-expert scales, biases, shared experts, or
LoRA contributions. A first prototype must select one supported architecture
and reproduce that architecture's exact order of operations.

### 4. Transfer and scheduling behavior

The generic scheduler is created by `llama_context::sched_reserve()` and
allocates/executes graphs through
`ggml_backend_sched_alloc_graph()` and
`ggml_backend_sched_graph_compute_async()` at
[`src/llama-context.cpp:1368`](llama.cpp/src/llama-context.cpp) and
[`src/llama-context.cpp:2494`](llama.cpp/src/llama-context.cpp).

`ggml_backend_sched_split_graph()` assigns backend IDs to graph nodes and
propagates those assignments across adjacent operations in
[`ggml/src/ggml-backend.cpp:1055`](llama.cpp/ggml/src/ggml-backend.cpp). The public
override `ggml_backend_sched_set_tensor_backend()` is per tensor/node
([`ggml/include/ggml-backend.h:336`](llama.cpp/ggml/include/ggml-backend.h)); it
cannot assign expert 0 to one backend and expert 16 to another within one
`MUL_MAT_ID` weight tensor.

There is one relevant optimization in
`ggml_backend_sched_compute_splits()` at
[`ggml/src/ggml-backend.cpp:1641`](llama.cpp/ggml/src/ggml-backend.cpp): when
MoE weights are host-resident, it reads IDs, finds used experts, and copies only
consecutive used expert slices to the backend executing the `MUL_MAT_ID`. This
reduces host-to-device bytes, but the selected slices are still copied to the
single backend running the operation. It is dynamic weight staging, not
stationary multi-GPU expert placement.

## SGLang reference structure

SGLang separates the concerns that llama.cpp currently fuses into one graph
operation:

1. `DeepEPMoE.forward_impl()` calls dispatcher `dispatch()`, runs the local MoE
   core, then calls `combine()` in
   [`python/sglang/srt/layers/moe/ep_moe/layer.py:250`](sglang/python/sglang/srt/layers/moe/ep_moe/layer.py).
2. `StandardDispatcher` maintains `moe_ep_size`, a local-expert count, and a
   `local_expert_mapping`; it maps global top-k IDs to local IDs before the
   local expert runner at
   [`python/sglang/srt/layers/moe/token_dispatcher/standard.py:98`](sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py).
3. DeepEP's normal path computes token counts per rank/expert, dispatches
   activations and routing metadata, executes local experts, and combines output
   back. The dispatch core is
   [`python/sglang/srt/layers/moe/token_dispatcher/deepep.py:563`](sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py); the return path is
   `_combine_core()` at approximately line 647.
4. `ExpertLocationMetadata` explicitly stores physical-to-logical mappings,
   logical-to-physical mappings, `ep_size`, and local physical expert counts in
   [`python/sglang/srt/eplb/expert_location.py:58`](sglang/python/sglang/srt/eplb/expert_location.py).
   `init_trivial()`/`init_by_mapping()` provide static mapping; `init_by_eplb()`
   adds measured-load-based remapping.
5. SGLang's load balancing is downstream of observation: expert distribution
   recording counts selected experts and aggregates them to GPU utilization in
   [`python/sglang/srt/eplb/expert_distribution.py:compute_gpu_physical_count`](sglang/python/sglang/srt/eplb/expert_distribution.py).

For this project, the useful reference is the ownership/dispatch/combine split,
not DeepEP itself. PCIe x1 is a reason to avoid adding its collectives to the
first prototype.

## PCIe activation traffic

Let:

```text
H       = routed expert input width in elements (usually n_embd)
R       = routed expert output width in elements (usually n_embd)
K       = experts selected per token (n_expert_used)
b_in    = bytes per input activation element
b_out   = bytes per output activation element
m_t     = number of selected expert assignments for token t whose owner is remote
u_t     = number of distinct remote owner GPUs selected for token t
```

With owner grouping, one input vector is needed per remote owner. Each owner can
also apply the routing weights and sum its selected local experts before sending
one partial output vector back. The minimum logical activation payload for token
`t` is therefore:

```text
bytes_to_remote(t)   = u_t * H * b_in
bytes_from_remote(t) = u_t * R * b_out
logical_payload(t)   = u_t * (H*b_in + R*b_out)
```

If the remote GPU instead returns every expert result separately, the payload is
`u_t*H*b_in + m_t*R*b_out`; this is a valid but non-minimal implementation.

For the existing CUDA `MUL_MAT_ID` path, the routed input and result are F32:
`ggml_mul_mat_id()` creates an F32 result, and the CUDA MoE vector path requires
F32 input/output. Thus, before an explicit communication cast/quantization
policy is introduced, equal-width execution has a baseline logical payload of
`u_t * 8H` bytes per token. FP16/BF16 payloads (`u_t * 4H`) are possible only if
the prototype deliberately defines and validates a lower-precision transfer
format.

For a batch, sum these quantities across tokens. Routing metadata remains small
but required: `(token_index, slot_index, local_expert_id)` must preserve output
order, and the owner needs a routing weight if it combines before return. The
origin may retain the original global IDs and weights; only the owner-local ID
and the data necessary for its selected computation must cross the boundary.

`logical_payload` is not a PCIe DMA counter. The prototype must report logical
payload separately from transport bytes and time. CUDA's direct peer-copy path
uses `cudaMemcpyPeerAsync`; if peer access is unavailable on the CMP/P104
topology, an explicit host-staged fallback changes the observed D2H/H2D traffic
and synchronization cost. Peer-copy availability and measured bandwidth are
therefore mandatory startup preflight checks.

## Current same-device assumptions

The assumptions that must be broken or isolated are:

- One repeating layer has one `dev_layer[il]` and one buffer-list selection.
- A merged expert tensor's expert dimension is a local indexing dimension, not a
  placement dimension.
- `MUL_MAT_ID` consumes the complete weight tensor and selected IDs on one
  backend.
- Once experts are packed per owner, router IDs must be translated from global
  expert IDs to that owner's local tensor indices before `MUL_MAT_ID` can use
  them.
- Gate/up, activation, down, per-expert scales/biases, and the reduction are
  formed as one graph region whose tensors are backend-assigned as whole objects.
- Scheduler weight preference is based on the buffer containing the tensor; it
  does not inspect `ids` to select among GPU owners.
- Existing selective expert copying assumes the source is host memory and the
  destination is the one backend executing the split.

These are structural, not merely command-line limitations.

## Smallest prototype

Target: two GPUs, one MoE layer, manually configure
`expert_to_owner[e] = e < 16 ? GPU0 : GPU1`, no migration or replication.
For GPU1's packed 16-expert tensor, additionally configure
`expert_to_local[e] = e - 16`; GPU0 has `expert_to_local[e] = e`.

Recommended prototype phases:

1. Select one MoE architecture with post-FFN routing weights, and explicitly
   exclude Llama 4's pre-FFN weighting, routed-expert LoRA, and unsupported
   per-expert bias/scale/shared-expert variants from this first test.
2. Add an explicit graph boundary on GPU0 after the router/top-k and the
   pre-expert activation. Materialize the IDs, weights, and input activation;
   later inject the combined `moe_out` into a suffix graph. This is necessary
   because the normal llama.cpp graph schedules the full layer as one DAG.
3. Partition selected assignments by `expert_to_owner[]`, remap global IDs to
   `expert_to_local[]`, and retain `(token_index, slot_index)` metadata. Store
   each owner's stationary weights in a local packed expert tensor.
4. For every remote owner, coalesce its selected token rows and copy each input
   activation once. First use direct CUDA peer copy only after a P2P capability
   test; otherwise use an explicit, measured host-staged path.
5. Run owner-local gate/up, activation, and down operations. Apply routing
   weights and sum all selected local experts per token on the owner, then return
   one owner partial result per token to GPU0.
6. Sum owner partials on GPU0 and compare with the unmodified single-GPU
   result using an architecture-appropriate numerical tolerance. Parallel owner
   reduction can change floating-point addition order, so bitwise equality is
   not a valid general criterion.
7. Count logical payload from `u_t`, and independently record copy API bytes,
   P2P versus host-staged traffic, synchronization, and GEMM time.

This can be implemented first as a deliberately explicit executor around one
layer. It avoids pretending that a single `ggml_tensor` can be split across
backends while preserving the existing op contract.

## Files/functions likely to change after the harness succeeds

- `src/llama-model.h/.cpp`: add static expert placement metadata and create/load
  per-device packed expert slices plus global-to-owner/local maps; include scale
  and bias tensors where applicable.
- `src/llama-graph.cpp`: add a static-placement MoE graph path or a custom node
  boundary around router, owner dispatch, and combine, including explicit prefix
  and suffix graph inputs/outputs for the first harness. The existing
  construction points are `llm_build_moe_ffn()` and `build_lora_mm_id()`.
- `ggml/include/ggml.h`, `ggml/src/ggml.c`: if graph-native, define an operation
  or API carrying owner IDs/grouped activation slots. Do not overload the current
  `MUL_MAT_ID` contract without a backend-wide implementation plan.
- `ggml/src/ggml-backend.cpp`: add explicit peer-or-host-staged activation copy
  scheduling, owner-group buffers, P2P preflight, and separate logical/DMA byte
  counters. The existing selective host expert copy is the closest transfer
  precedent, but must not be repurposed as GPU weight migration.
- GPU backend implementations of `MUL_MAT_ID` or the new operation: execute
  only the local expert slice and accept grouped token rows.
- `src/llama-context.cpp`: allocate/reuse per-owner activation buffers and
  synchronize copies/GEMMs in a deterministic order suitable for PCIe x1.
- `tests/test-backend-ops.cpp` and a llama integration test: verify exact output
  against one-device execution, then verify measured bytes against the formula.

## Recommended acceptance criteria

The first milestone is complete only when a two-GPU, one-MoE-layer run has:

- stationary expert weights, with experts 0-15 on GPU0 and 16-31 on GPU1;
- numerically correct output against the existing single-device path within a
  stated tolerance and for one explicitly supported MoE architecture;
- logical remote activation payload equal to the sum of
  `u_t*(H*b_in + R*b_out)`, plus separately reported routing metadata;
- direct-peer versus host-staged transport bytes and time reported separately,
  with P2P capability recorded before the run;
- no expert weight movement during generation;
- a trace showing copy, owner GEMM, return copy, and combine ordering.

Only after this should the implementation be moved into the generic graph and
evaluated on the CMP/P104 topology.
