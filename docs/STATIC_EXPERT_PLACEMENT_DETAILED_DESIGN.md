# Detailed Design: Static MoE Expert Placement in llama.cpp

## 1. Status

This document is an implementation-oriented design for a deliberately narrow
prototype. It incorporates the findings from a fresh review of:

- llama.cpp at commit `0a2cee446`;
- SGLang at commit `61fa64ae7e`;
- the existing `STATIC_EXPERT_PLACEMENT_DESIGN.md` note; and
- the two-GPU, PCIe x1 milestone described below.

> REMARK: IMPLEMENTATION STATUS. The repository currently contains the opt-in
> public parameter structure, the isolated `llama-static-expert` mapping and
> validation sidecar, model-load-time construction, and focused validation
> tests. Passing enabled static-placement parameters currently validates the
> configuration and then fails explicitly because expert tensor sharding,
> prefix/suffix graph interception, and runtime activation dispatch are not yet
> complete. This prevents an accidental silent fallback to the existing path.

Source line numbers are references to those revisions and may move as either
repository changes.

### Merge annotation convention

> REMARK: PROPOSED ADDITION. Every new design or code path described below is
> explicitly marked with `REMARK:`. These annotations identify changes that can
> be kept isolated while rebasing onto newer llama.cpp revisions. Existing
> llama.cpp behavior is described without that marker.

## 2. Design question

What is the smallest correct change that lets llama.cpp keep MoE expert weights
stationary on heterogeneous GPUs while moving only routed activations and
results across a slow PCIe topology?

The first milestone is:

> On two GPUs, statically place experts 0-15 of one 32-expert MoE layer on GPU0
> and experts 16-31 on GPU1, produce numerically correct decode output, and
> measure the bytes and latency introduced by remote expert execution.

The first implementation is a measurement harness, not a general expert
parallel runtime.

## 2A. Compatibility and opt-in contract

Backward compatibility is a hard requirement, not a follow-up task.

> REMARK: PROPOSED ADDITION. When static placement is disabled, model loading,
> `llama_layer` expert tensor fields, GGUF tensor names, `MUL_MAT_ID`, graph
> construction, scheduler placement, warmup, and all existing MoE execution
> paths must remain unchanged.

The default value must be disabled:

```cpp
enum llama_static_expert_dispatch_mode {
    LLAMA_STATIC_EXPERT_DISPATCH_HOST = 0,
    LLAMA_STATIC_EXPERT_DISPATCH_FIXED_GPU = 1, // Reserved; future design.
};

struct llama_static_expert_params {
    bool enabled = false;
    int32_t target_layer = -1;
    int32_t owner_count = 0;
    const int32_t * expert_to_owner = nullptr;
    int32_t expert_to_owner_count = 0;
    bool require_peer_copy = false;
    llama_static_expert_dispatch_mode dispatch_mode =
        LLAMA_STATIC_EXPERT_DISPATCH_HOST;
};
```

> REMARK: PROPOSED ADDITION. `LLAMA_STATIC_EXPERT_DISPATCH_HOST` is the only
> implemented mode in this document. Reserve
> `LLAMA_STATIC_EXPERT_DISPATCH_FIXED_GPU` for a future design and reject it as
> unsupported until that implementation exists. Do not silently fall back to a
> different mode when an explicitly requested mode is unavailable.

> REMARK: PROPOSED ADDITION. Add this configuration at the model-loading
> boundary, because owner-packed tensors must be selected before persistent
> model weight allocation. Append an opt-in field to the existing model
> parameter structure, or expose an equivalent new constructor/configuration
> API. Do not reinterpret existing split-mode or tensor-split parameters.

The activation rule is:

```text
static_expert.enabled == false
  -> execute the current llama.cpp path byte-for-byte in structure

static_expert.enabled == true
  -> validate the new configuration
  -> construct the sidecar placement object
  -> use owner-packed tensors only for the configured target layer
```

No automatic activation is allowed based on the number of GPUs, GPU type,
environment variables, or a detected MoE model. An invalid opt-in configuration
must fail clearly; an omitted configuration must select the old path.

For the phase-one host-dispatch mode, the existing model router remains the
router. Static placement does not replace router logits, top-k selection, or
routing-weight calculation. The new phase-one code performs owner lookup,
local-ID remapping, dispatch, owner-side reduction, and transfer orchestration
after the existing router has produced its outputs.

The mode is still selected early: model loading creates owner-packed tensors
and graph construction selects the alternate target-layer execution boundary.
It is only the route-dependent owner decision that occurs after top-k at
runtime. Those additions are entered only when the new parameters enable the
sidecar.

> REMARK: FUTURE DESIGN. A later fixed-GPU mode may replace the configured
> layer's complete router-to-combine graph region with an integrated GPU
> implementation. It must reuse the same router weights and exactly preserve
> router logits, masking, top-k selection, normalization, and routing-weight
> semantics. It is a new execution path, not a new routing policy.

The default path must also avoid allocating empty placement maps, owner buffers,
temporary loader state, dispatch queues, or new graph nodes. This keeps the
normal model's memory footprint and scheduler behavior unchanged.

> REMARK: PROPOSED ADDITION. Add an explicit regression test that constructs a
> model with the normal default parameter initializer, runs an MoE layer, and
> verifies that the static sidecar is null and the existing expert tensors are
> used. A second test passes the new parameters and verifies that the sidecar is
> non-null. This proves activation is controlled by the parameter, not by
> hardware discovery.

The public/API-facing configuration should be append-only and initialized by the
existing default-parameter function. Callers that do not know about static
placement continue to compile and receive `enabled == false`. CLI or test-only
options may populate the structure, but the runtime must not silently enable it
from an environment variable.

## 2B. Isolation strategy and proposed new class

> REMARK: PROPOSED ADDITION. Put new state and orchestration in a dedicated
> sidecar rather than adding owner vectors and runtime buffers directly to
> `llama_layer` or changing the meaning of existing expert tensor fields.

Proposed files:

```text
src/llama-static-expert.h
src/llama-static-expert.cpp
```

The sidecar can contain:

```cpp
class llama_static_expert_placement {
public:
    static std::unique_ptr<llama_static_expert_placement> create(
        const llama_static_expert_params & params,
        const llama_model & model,
        const llama_model_loader & loader);

    bool enabled() const;
    bool handles_layer(int32_t layer) const;

    // Initialization-time owner tensor construction and validation.
    bool prepare_layer(llama_layer & layer);

    // Runtime prefix/dispatch/owner-compute/combine orchestration.
    bool execute(llama_context & context, int32_t layer, ...);
};
```

The exact signatures may follow local llama.cpp conventions, but ownership of
the following concerns should remain in this class:

- layer-indexed owner/local maps;
- validation of the opt-in configuration;
- temporary-source weight sharding;
- persistent owner tensor references;
- reusable activation and metadata staging buffers;
- P2P/host-staged transfer selection;
- counters and trace events; and
- owner dispatch and reduction.

The existing model and context objects should hold only an optional pointer or
`unique_ptr` to this sidecar. They should not gain new per-expert fields unless
the selected implementation proves that an existing ownership boundary cannot
express the required reference.

This structure limits merge conflicts: upstream changes to normal expert tensor
creation and generic graph scheduling remain untouched when the feature is
disabled, while the opt-in integration points are small and easy to re-audit.

## 3. Scope

### 3.1 Included

- Two CUDA devices.
- One explicitly configured MoE layer.
- Exactly 32 routed experts split 16/16 for the initial milestone.
- Decode with `n_tokens == 1`.
- Static, layer-indexed expert ownership.
- Stationary packed expert weights after model initialization.
- Global-to-owner and global-to-owner-local ID translation.
- Owner-grouped activation dispatch.
- Owner-local gate/up, activation, down, routing-weight application, and sum.
- One partial result returned per remote owner per token.
- Direct CUDA peer copy when it is enabled and operational.
- An explicit, measured host-staged fallback.
- Correctness comparison and detailed traffic/timing counters.

### 3.2 Excluded

- Tensor parallelism and `LLAMA_SPLIT_MODE_ROW`.
- Pipeline parallelism as a performance strategy.
- Dynamic expert migration.
- Expert replication.
- Cache migration.
- NCCL, DeepEP, all-to-all, or other collectives.
- Automatic load balancing.
- Multi-token prefill.
- Multi-node execution.
- Llama 4 pre-FFN routing-weight semantics.
- Routed-expert LoRA in the first prototype.
- A generic graph-native expert dispatch operation.
- Integrated GPU-side router mapping and fixed-owner dispatch. This is retained
  as a future design in section 9.6.
- Communication quantization in the baseline.

## 4. Required invariants

The prototype is correct only if all of these invariants hold:

1. Every routed expert has exactly one owner for the configured layer.
2. Every global expert ID maps to a valid owner-local expert ID.
3. Gate, up, down, scales, and biases for one expert have the same owner.
4. Expert weights do not move after initialization and warmup.
5. The router remains authoritative for global expert IDs.
6. Remote execution preserves the model's existing operation ordering.
7. Routing weights are applied exactly once.
8. A token activation crosses to a remote owner at most once.
9. A remote owner returns at most one hidden-width partial per token.
10. The baseline and distributed paths use the same model weights and inputs.
11. Logical payload, API-requested copy bytes, and observed transport are
    reported as different measurements.

## 5. Current llama.cpp implementation

### 5.1 Expert representation

MoE projection weights are generally represented as three-dimensional tensors
whose third dimension is expert ID:

```text
gate_up_exps = [n_embd, 2*n_ff, n_expert]
gate_exps    = [n_embd,   n_ff, n_expert]
up_exps      = [n_embd,   n_ff, n_expert]
down_exps    = [  n_ff, n_embd, n_expert]
```

Names are defined as `blk.%d.ffn_{gate,up,down}_exps` in
`llama.cpp/src/llama-arch.cpp`. The common helper
`llama_model_base::create_tensor_gate_up_exps()` in
`llama.cpp/src/llama-model.cpp:2931` creates merged gate/up or separate gate and
up tensors. Down tensors and architecture-specific auxiliary tensors are
created by the individual implementations under `llama.cpp/src/models/`.

This distinction matters: sharding only the common helper is not sufficient.
The selected architecture must enumerate every expert-indexed tensor used by
its routed FFN.

### 5.2 Layer placement

`llama_model_base::load_tensors()` builds one `dev_layer[il]` placement record
per repeating layer. Under layer split mode, `get_layer_buft_list()` selects the
device and buffer types used by complete tensors in that layer.

The scheduler can prefer the backend containing a weight tensor, but its unit
of placement is a complete `ggml_tensor`. A single tensor cannot assign expert
slices to different backends.

The current effective model is:

```text
layer -> buffer list -> complete expert tensor -> one backend for MUL_MAT_ID
```

### 5.3 Router and expert call path

The routed FFN is built by
`llm_graph_context::build_moe_ffn()` in
`llama.cpp/src/llama-graph.cpp:1917`.

The relevant path is:

```text
input activation
  -> router projection
  -> probabilities and optional routing transformations
  -> top-k global expert IDs
  -> selected routing weights
  -> gate/up MUL_MAT_ID
  -> activation
  -> down MUL_MAT_ID
  -> routing-weight application
  -> expert-slot reduction
  -> moe_out
```

For most supported architectures, routing weights are applied after the down
projection. Llama 4 is a material exception: `weight_before_ffn` at
`llama.cpp/src/llama-graph.cpp:1944` repeats and weights the input before expert
execution. The first prototype must use an architecture with post-FFN weights.

### 5.4 MUL_MAT_ID contract

`ggml_mul_mat_id()` in `llama.cpp/ggml/src/ggml.c:3318` has this logical
contract:

```text
as  = [cols, rows, n_expert]
b   = [cols, n_selected, n_tokens]
ids = [n_selected, n_tokens]
c   = [rows, n_selected, n_tokens]
```

`ids` selects slices of `as`. There is no owner argument. The operator expects
all selectable slices in one weight tensor and executes the node on one
backend.

The prototype should reuse this contract independently on each owner with
owner-local packed tensors and owner-local IDs. It should not change the public
operator contract in phase one.

### 5.5 Scheduler and transfers

`llama_context::process_ubatch()` builds or reuses one graph, allocates it with
`ggml_backend_sched_alloc_graph()`, sets graph inputs, and computes it through
the backend scheduler. Graph reuse requires stable topology, as noted in
`llama.cpp/src/llama-context.cpp:1335`.

Backend tensor copies require identical source and destination layouts:

```text
ggml_backend_tensor_copy()
ggml_backend_tensor_copy_async()
```

Both assert `ggml_are_same_layout()` in
`llama.cpp/ggml/src/ggml-backend.cpp:477` and line 500. Arbitrary selected rows
therefore require explicit contiguous staging tensors. They cannot be copied as
unrelated strided views.

The CUDA async path uses `cudaMemcpyPeerAsync()` for distinct physical devices.
Peer access is enabled during CUDA initialization only when `GGML_CUDA_P2P` is
set (`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:392`). Capability detection
alone does not prove that the intended runtime path is active.

### 5.6 Existing selective expert staging

`ggml_backend_sched_compute_splits()` contains an optimization for host-resident
MoE weights. It reads routed IDs and copies consecutive used expert slices to
the backend executing `MUL_MAT_ID`.

That path is useful as a reference for expert-slice offsets and contiguous
grouping, but it implements dynamic weight movement to one executor. It must
not be reused as the stationary ownership mechanism.

### 5.7 Warmup behavior

During warmup, `llm_graph_context` uses `n_expert` instead of normal
`n_expert_used` (`llama.cpp/src/llama-graph.cpp:1445`). A design sized only for
normal top-k decode will fail during warmup.

Phase one will bypass the static executor during standard model warmup and run a
separate static-executor initialization pass after owner buffers are allocated.
The measured generation interval begins after both warmup paths complete.

## 6. SGLang concepts retained

SGLang separates responsibilities that llama.cpp currently combines:

```text
logical routing
  -> logical-to-physical placement
  -> rank ownership and rank-local ID
  -> activation dispatch
  -> local expert execution
  -> combine
```

`ExpertLocationMetadata` in
`sglang/python/sglang/srt/eplb/expert_location.py:58` stores mappings with a
layer dimension. Dispatchers convert IDs before local expert execution, and the
DeepEP path explicitly separates dispatch from combine.

The llama.cpp prototype retains these ideas:

- placement is layer-indexed;
- router IDs stay global;
- execution IDs are owner-local;
- dispatch and combine are explicit phases; and
- placement policy is separate from execution.

It does not adopt replicas, dynamic physical-slot remapping, load balancing, or
DeepEP communication.

## 7. Proposed data model

Expert IDs are layer-local. The minimum safe metadata is therefore:

```cpp
struct llama_expert_placement {
    int32_t layer;
    int32_t n_expert;
    int32_t n_owner;

    // Indexed by global expert ID within this layer.
    std::vector<int16_t> expert_to_owner;
    std::vector<int16_t> expert_to_local;

    // Indexed by owner, then owner-local expert ID.
    std::vector<std::vector<int16_t>> local_to_expert;
};
```

The model-level container is conceptually:

```cpp
std::vector<std::optional<llama_expert_placement>> expert_placement_by_layer;
```

The first configuration is generated manually:

```text
target_layer = configured layer index

expert  0-15 -> owner 0, local ID  0-15
expert 16-31 -> owner 1, local ID  0-15
```

Validation at model load must reject:

- an out-of-range layer or expert ID;
- a missing or duplicate expert assignment;
- an unavailable owner device;
- a non-contiguous local ID range;
- inconsistent dimensions among routed expert tensors; and
- row-split mode.

The maps are runtime configuration in phase one, not new GGUF metadata.

## 8. Stationary weight layout and loading

### 8.1 Required owner tensors

For the configured layer, each owner receives packed tensors containing only
its experts. For a 16/16 split:

```text
owner0_gate_up = [n_embd, 2*n_ff, 16]
owner0_down    = [n_ff, n_embd, 16]
owner1_gate_up = [n_embd, 2*n_ff, 16]
owner1_down    = [n_ff, n_embd, 16]
```

Separate gate/up architectures use separate owner tensors. Any expert-indexed
scale or bias tensor must be packed using the same local ordering. Shared
experts remain on the layer's normal device and are out of the dispatched
routed-expert path.

### 8.2 Loader constraint

The existing loader cannot request a smaller shape for one GGUF tensor:

- `llama_model_loader::create_tensor()` checks source dimensions and asserts
  equal byte count at `llama.cpp/src/llama-model-loader.cpp:1299`.
- `llama_model_loader::load_data_for()` loads the complete named tensor.
- `TENSOR_DUPLICATED` duplicates a complete tensor; it is not slice loading.

### 8.3 Phase-one loading decision

Use temporary-source load-then-scatter initialization. The original full tensor
must not be allocated as a persistent tensor in the candidate model's normal
weight buffer. Model weight tensors share larger backend allocations, so an
individual full tensor cannot be copied and then freed independently.

> REMARK: PROPOSED ADDITION. This alternate loading path is entered only after
> `static_expert.enabled` has been validated. The existing loader code remains
> the only path when the flag is false.

1. During target-layer tensor creation, create owner-packed persistent tensors
   instead of the normal full expert tensor.
2. Retain the GGUF source tensor metadata and file/mmap byte range under its
   original name.
3. Expose the full source through mmap when available, or load it into a
   dedicated temporary host backend buffer. The temporary buffer must have an
   independent lifetime from persistent model buffers.
4. Allocate packed owner tensors on their final devices.
5. Copy each contiguous expert-dimension range from the source or temporary
   host tensor into its owner tensor.
6. Validate byte offsets, tensor type, strides, quantization block alignment,
   and destination sizes.
7. Synchronize all owner uploads, then release the entire temporary host buffer.

This requires a loader API that can identify and read a GGUF tensor byte range
without first registering a same-shaped persistent model tensor. Reusing
`load_data_for()` unchanged is insufficient because it resolves by destination
tensor name and reads `ggml_nbytes(destination)`.

The manual 0-15/16-31 placement intentionally makes each owner slice contiguous
in dimension 2. This avoids a general gather during initialization.

The implementation must not assume that quantized data can be repacked by
element count. It must use `nb[2]` and byte ranges derived from the source
tensor layout. Every owner starts at an expert-slice boundary, so the quantized
rows inside each expert remain unchanged.

### 8.4 Later loader improvement

After the prototype succeeds, add fully streamed shard-aware GGUF loading that
writes bounded chunks directly to owner upload buffers. That removes the
temporary full host tensor when mmap is unavailable and reduces peak memory.
It is not required for initial correctness, but the phase-one path must expose
peak temporary memory in its diagnostics.

### 8.5 Meaning of stationary

Weight movement is permitted only during model initialization. Counters for the
acceptance test start after:

- owner tensor allocation;
- slice copies;
- copy synchronization;
- owner graph allocation; and
- warmup.

Any expert-weight transfer recorded after this point fails the milestone.

## 9. Phase-one execution architecture

### 9.1 Target-layer interception

Static placement is enabled for exactly one configured layer. All other layers,
including other MoE layers, execute through the unmodified graph path.

> REMARK: PROPOSED ADDITION. The model builder should branch on
> `static_expert_placement != nullptr && handles_layer(layer)`. The false branch
> must call the existing `build_moe_ffn()` code without changed tensor shapes or
> changed scheduler hints.

For the target layer, execution is split into:

```text
prefix graph on GPU0
  -> router/top-k outputs and pre-expert activation
  -> host dispatch decision
  -> owner-local expert graphs
  -> owner partials on GPU0
  -> combine
  -> suffix graph resumes normal layer placement
```

This is intentionally explicit. It avoids claiming that one ggml graph node or
weight tensor can have multiple owners.

The split is not available through an existing public llama.cpp API. For the
selected architecture, its model graph builder must gain an internal
continuation boundary with two modes:

```text
build_target_prefix:
  build normal layers through target-layer attention and routing
  expose pre-expert activation, selected IDs, and routing weights
  stop before routed expert projections

build_target_suffix:
  accept externally produced moe_out as an input tensor
  perform the target architecture's residual/shared-expert/post-FFN work
  continue through all later layers and model output
```

The continuation state must explicitly carry every live target-layer value
needed after the routed FFN. It must not recompute target-layer attention or
apply memory/KV updates twice. Phase two is restricted to one architecture so
this boundary can be implemented and audited in its `src/models/<arch>.cpp`
builder before attempting a common abstraction.

### 9.2 Host-visible versus GPU-resident data

Only the small routing information required to choose owners and populate
fixed-size owner inputs becomes host-visible in phase one:

```text
Host-visible:
  selected global expert IDs
  selected routing weights

GPU-resident:
  pre-expert activation
  packed owner inputs
  owner results
  combined moe_out
```

The activation must never make an origin-GPU-to-host round trip merely to make
the dispatch decision. It remains in a contiguous GPU0 staging tensor and is
copied GPU-to-GPU or through the explicitly measured fallback.

Reading IDs and weights on the host introduces one synchronization barrier after
routing. The host then uploads each active owner's `K` local IDs and `K` padded
weights. This small control transfer is counted as routing metadata, not
activation payload. The barrier and uploads are part of phase-one latency and
must be timed separately.

### 9.3 Fixed owner graph shapes

The number of assignments for one owner varies from zero to `K`. Rebuilding a
graph for every routing pattern would defeat graph reuse and contaminate timing.

Each owner therefore gets a fixed `K`-slot graph:

```text
owner_input   = [H, 1, 1]
owner_ids     = [K, 1]
owner_weights = [K, 1]
owner_partial = [R, 1]
```

Assignments owned by that GPU occupy the first `k_owner` slots. Remaining slots
are padded with a valid local expert ID and zero routing weight. The padded
expert may execute, but contributes zero to the partial. If `k_owner == 0`, the
owner graph is skipped and its partial is treated as zero.

This wastes some expert computation but gives stable graph topology and isolates
the communication experiment. Compact or cached graph variants are later
optimizations.

### 9.4 Decode dispatch sequence

For one token:

1. Execute the prefix graph on GPU0.
2. Synchronize once to read `K` selected global expert IDs and weights.
3. Validate each ID and group selected slots by owner.
4. Convert each global ID with
   `expert_to_local[target_layer][global_id]`.
5. Fill each active owner's fixed-size ID and routing-weight staging tensors.
6. For GPU0, bind the original activation directly to the owner0 graph.
7. For GPU1, copy the activation once into its `[H, 1, 1]` staging tensor.
8. Execute each active owner graph using its stationary packed weights.
9. Apply routing weights and sum all of that owner's selected slots locally.
10. Copy GPU1's one `[R, 1]` partial back to GPU0.
11. Add owner partials in a deterministic owner order.
12. Bind the combined `moe_out` to the suffix graph and continue evaluation.

GPU0 is the configured target layer's origin device, not a requirement that all
later layers execute there. Once the combined target-layer output is available,
the suffix follows existing whole-layer placement. Benchmark reporting must
separate ordinary later-layer transfers from static-expert dispatch traffic.

The owner graph reproduces the selected architecture's existing operation
ordering. It must not apply routing weights before the FFN for a model whose
baseline applies them after the down projection.

### 9.5 Routing metadata

For decode, owner execution requires:

```text
owner-local expert ID
routing weight
valid-slot count or zero-padded weight convention
```

The origin retains global IDs and original slot order for diagnostics. Token and
slot indices become necessary when prefill is added, but do not need to cross
the device boundary for a one-token owner-side reduction.

### 9.6 Performance limitation and future integrated dispatch

> REMARK: PROTOTYPE LIMITATION. Host dispatch introduces this serialized section
> on every token:

```text
GPU router
  -> synchronize router outputs
  -> CPU reads IDs and weights
  -> CPU maps owners and fills owner metadata
  -> launch activation copies and owner graphs
```

On PCIe x1, this synchronization and launch latency may exceed the time needed
to transfer the approximately 23 KiB equal-width F32 activation/result payload
for one remote owner at `H = 2880`. Phase one must measure this cost explicitly;
it must not be presented as the final performance architecture.

> REMARK: FUTURE DESIGN. `LLAMA_STATIC_EXPERT_DISPATCH_FIXED_GPU` should be
> selected during model loading and graph construction, just like host dispatch.
> For the configured layer it should build an alternate integrated path:

```text
router projection and top-k on origin GPU
  -> GPU global-ID-to-owner/local-ID mapping
  -> fixed owner inputs
  -> stationary owner expert execution
  -> owner reduction
  -> origin combine
```

The ordinary `build_moe_ffn()` path remains untouched and is used whenever
static placement is disabled. When fixed-GPU dispatch is enabled, the configured
layer does not enter the ordinary routed-expert region; the alternate region is
chosen before token execution.

There is an unresolved conditional-communication tradeoff that belongs in a
separate future implementation document:

```text
Fixed communication to every configured owner:
  no host owner-decision synchronization
  stable reusable graph
  may send an activation when that owner has no selected expert

Conditional communication after reading an owner-active flag:
  preserves minimum route-dependent payload
  still introduces a GPU-to-host decision or another synchronization mechanism

Fully GPU-controlled conditional cross-device dispatch:
  can avoid both costs
  requires a specialized CUDA/runtime communication mechanism beyond phase one
```

The future document must benchmark the host synchronization cost against the
extra fixed transfer cost on the actual CMP/P104 topology before choosing the
default performance mode. It must also specify how cross-device launches and
events are driven without reintroducing an implicit CPU barrier.

## 10. Transfer policy

### 10.1 Direct peer path

At startup, record for both directions:

- CUDA device IDs and physical device IDs;
- compute capability;
- `cudaDeviceCanAccessPeer()` result;
- whether `GGML_CUDA_P2P` is set;
- whether peer access was enabled;
- whether the backend async copy implementation accepted the copy; and
- measured one-way bandwidth and latency for the exact staging sizes.

Capability without enabled access is reported as unavailable for this test.

### 10.2 Host-staged fallback

If direct peer copy is unavailable or fails preflight:

1. Allocate reusable pinned host buffers.
2. Copy GPU0 activation to host.
3. Copy host activation to GPU1.
4. Reverse the process for the owner partial.
5. Synchronize with explicit events or backend synchronization points.

The fallback must be explicit. It must not infer transport behavior solely from
a successful `cudaMemcpyPeerAsync()` API call.

### 10.3 Buffer reuse

All activation, ID, weight, and partial staging buffers are allocated once and
reused. No per-token device allocations are allowed in measured generation.

## 11. Traffic model

Let:

```text
H       = routed expert input width in elements
R       = routed expert output width in elements
K       = selected experts per token
b_in    = bytes per transferred input element
b_out   = bytes per transferred output element
m_t     = remote expert assignments for token t
u_t     = distinct remote owners selected for token t
```

With one input per owner and owner-side weighted reduction:

```text
bytes_to_remote(t)   = u_t * H * b_in
bytes_from_remote(t) = u_t * R * b_out
logical_payload(t)   = u_t * (H*b_in + R*b_out)
```

If every remote expert result is returned separately, the non-minimal result is:

```text
u_t*H*b_in + m_t*R*b_out
```

The CUDA routed paths used by the baseline expect F32 activation/output for the
relevant quantized `MUL_MAT_ID` kernels, and `ggml_mul_mat_id()` produces F32.
For equal widths without a communication cast:

```text
logical_payload(t) = u_t * 8H bytes
```

For the GPT-OSS-sized test dimensions already represented in backend tests,
`H = R = 2880`, one remote owner requires:

```text
2880*4 + 2880*4 = 23,040 logical activation bytes per token
```

This is payload, not total PCIe traffic. A host-staged transfer performs two DMA
legs for each logical direction and must report each leg separately.

Routing metadata is reported separately. At minimum it includes local IDs,
routing weights, and any valid-slot count copied to an owner.

## 12. Instrumentation

The following counters are required per token and in aggregate:

```text
selected global expert IDs
owner chosen for each slot
k_owner for each owner
u_t and m_t
logical activation bytes sent
logical partial bytes returned
routing metadata bytes
requested D2D/P2P copy bytes
requested D2H bytes
requested H2D bytes
unexpected weight-copy bytes after warmup
```

Required timings are:

```text
router/prefix graph
router-ID readback synchronization
host grouping and remapping
activation copy per owner
owner gate/up
owner activation
owner down
owner weighting and reduction
partial return copy
origin combine
suffix graph
end-to-end target-layer latency
```

CUDA events should measure GPU work where possible. Host monotonic timers should
measure synchronization and control overhead. The report must state which clock
produced each timing.

## 13. Correctness and test strategy

### 13.1 Test A: synthetic two-backend executor

Create a focused multi-backend test independent of full model generation:

- deterministic synthetic F32 or Q4_0 expert tensors;
- 32 experts, top-k 4, hidden dimensions small enough for rapid testing;
- deterministic IDs covering local-only, remote-only, and mixed routing;
- fixed owner graphs with padding;
- direct comparison with one unsharded `MUL_MAT_ID` reference path.

This test validates packing, ID remapping, owner reduction, copies, and combine.
`tests/test-backend-ops.cpp` can continue testing individual operators, but its
current structure is not itself a multi-backend dispatch harness. Prefer a new
test executable or a clearly isolated multi-backend integration test.

### 13.2 Test B: one intercepted model layer

Choose one concrete model architecture and quantization before implementation.
Requirements are:

- 32 routed experts;
- post-FFN routing weights;
- no routed-expert LoRA;
- supported kernels on both target GPU generations;
- enough memory for each 16-expert owner slice; and
- a reference path that can run the target layer unsharded.

Enable static placement only on one selected layer. Capture the target layer's
input, router IDs, routing weights, baseline `moe_out`, and distributed
`moe_out`. Compare at the target-layer boundary before testing complete token
logits.

### 13.3 Test C: decode generation

Run deterministic greedy decode with:

- `n_tokens == 1` per evaluation;
- fixed prompt and seed;
- graph reuse enabled for stable owner graphs;
- static placement disabled for the baseline; and
- static placement enabled only for the target layer in the candidate run.

The baseline and candidate should be separate model instances or separate
processes loaded from the same GGUF. The candidate intentionally does not retain
the target layer's full unsharded expert tensors, and requiring both layouts in
one model would invalidate the stationary-memory measurement.

Compare target-layer output, final logits, and generated token IDs. State
absolute and relative tolerances. Bitwise equality is not required because
owner-side reduction can change floating-point addition order.

### 13.4 Required routing cases

Tests must force or locate tokens with:

- all selected experts on GPU0;
- all selected experts on GPU1;
- one remote and the remainder local;
- multiple selected experts on GPU1, proving one activation send and one return;
- repeated use of boundary experts 15 and 16; and
- zero assignments for one owner, proving that owner execution is skipped.

### 13.5 Failure tests

Reject or fail clearly when:

- only one CUDA device is present;
- row split mode is selected;
- the target layer is not a compatible MoE layer;
- the expert count is not 32 for the milestone configuration;
- owner tensors do not fit memory;
- a mapping is incomplete or invalid;
- packed byte ranges exceed the source tensor;
- an owner kernel does not support the selected weight type; or
- P2P is requested as mandatory but is not operational.

## 14. Implementation phases

### Phase 0: preflight utility

- Enumerate devices and relevant backend properties.
- Test direct copies in both directions at activation and result sizes.
- Test and report the pinned-host fallback.
- Produce bandwidth, latency, and path diagnostics.

Exit criterion: the transfer path is known before model changes are tested.

> REMARK: PROPOSED ADDITION. Phase 0 may be built and run independently of the
> model changes. It must not alter ordinary model initialization or graph
> execution.

### Phase 1: synthetic executor

- Implement layer-indexed maps and validation.
- Allocate packed owner weights and fixed-size staging buffers.
- Execute two owner-local MoE paths.
- Compare with an unsharded reference.
- Validate logical and requested-copy counters.

Exit criterion: all deterministic routing cases pass on two CUDA devices.

### Phase 2: model loading and one-layer interception

- Add load-then-scatter owner weight initialization.
- Intercept one configured compatible layer.
- Split prefix, owner execution, combine, and suffix.
- Keep all other layers on their existing path.
- Bypass static placement during standard warmup.

Exit criterion: target-layer output matches the original path within tolerance.

### Phase 3: decode benchmark

- Run deterministic token generation.
- Record per-token routes, payload, transport path, and timings.
- Prove no post-warmup expert-weight movement.
- Compare direct peer and explicit host-staged modes when both are available.

Exit criterion: complete acceptance criteria in section 17 are satisfied.

### Phase 4: later optimizations

Only after phase 3:

- direct shard-aware GGUF loading;
- compact owner graphs or graph variants;
- a separate detailed design for integrated GPU router mapping and fixed-owner
  dispatch, including its conditional-communication policy;
- GPU-side owner grouping to remove ID readback synchronization;
- prefill packing and token-index metadata;
- more than two owners;
- heterogeneous capacity-aware static placement;
- graph-native dispatch/combine operations; and
- optional lower-precision communication.

## 15. Expected file changes

### Required for the prototype

> REMARK: PROPOSED ADDITION. The following are new or changed files. Keep each
> change behind the opt-in check and avoid broad edits to the ordinary MoE path.

- `llama.cpp/src/llama-static-expert.h/.cpp`
  - New sidecar class described in section 2B.
  - New owner mapping, loading, dispatch, transfer, and instrumentation logic.

- `llama.cpp/src/llama-model.h`
  - Layer-indexed ownership metadata.
  - Owner-packed tensor references.

- `llama.cpp/src/llama-model.cpp`
  - Configuration validation.
  - Optional sidecar construction and owner tensor allocation.
  - Architecture-specific enumeration hooks.

- `llama.cpp/src/llama-model-loader.h`
- `llama.cpp/src/llama-model-loader.cpp`
  - Expert-slice initialization support or a controlled load-then-scatter hook,
    used only by the sidecar.
  - Source byte-range validation.

- `llama.cpp/src/llama-graph.h`
- `llama.cpp/src/llama-graph.cpp`
  - Expose the target layer's pre-expert activation, global IDs, and weights.
  - Build fixed-shape owner-local MoE graphs.
  - Accept combined `moe_out` as a suffix input.

- `llama.cpp/src/models/<selected-architecture>.cpp`
  - Add the architecture-specific prefix stop and suffix continuation point.
  - Preserve residual, shared-expert, scale, and post-FFN ordering.

- `llama.cpp/src/llama-context.h`
- `llama.cpp/src/llama-context.cpp`
  - Delegate to the optional sidecar for persistent staging buffers and host
    dispatcher state.
  - Orchestrate prefix, dispatch, owner compute, combine, and suffix.
  - Maintain counters and synchronization.

- `llama.cpp/ggml/src/ggml-backend.cpp`
  - Reuse existing copy APIs initially.
  - Add instrumentation or explicit fallback helpers only where public APIs are
    insufficient.

- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`
  - Expose or log enough information to prove which copy path executed.
  - Avoid changing `MUL_MAT_ID` semantics in phase one.

- `llama.cpp/tests/`
  - New two-backend static expert executor test.
  - Target-layer integration test.
  - Byte-accounting and failure-path tests.

> REMARK: PROPOSED ADDITION. The normal files and functions listed above must
> remain the default implementation. Do not replace `llama_layer` expert fields,
> rename GGUF tensors, or modify generic `MUL_MAT_ID` semantics merely to make
> the opt-in prototype compile.

### Not required initially

- `llama.cpp/ggml/include/ggml.h`
- `llama.cpp/ggml/src/ggml.c`

These become necessary only if a graph-native dispatch operation is introduced.
The first prototype can compose existing operations in owner-local graphs.

## 16. Key risks and mitigations

### Loader peak memory

Risk: retaining the original expert tensor while owner copies exist can exceed
VRAM or host memory.

Mitigation: load through host-accessible storage where possible, copy one owner
range at a time, release the full tensor immediately after synchronization, and
report peak temporary bytes.

### Heterogeneous kernel support

Risk: a quantized `MUL_MAT_ID` path supported on one GPU may fall back, reject,
or synchronize on the other.

Mitigation: preflight the exact tensor type and dimensions on both devices. Use
the synthetic Q4_0/F32 case before choosing the full model quantization.

### Host dispatch latency

Risk: reading IDs every token serializes the pipeline and dominates PCIe payload
time.

Mitigation: treat it as a measured prototype cost. A later GPU grouping kernel
can remove it after correctness is established.

### Fixed-shape wasted compute

Risk: zero-weight padding executes experts that were not selected for an owner.

Mitigation: report padded slots and wasted owner work. Do not optimize until the
fixed-shape implementation establishes a stable baseline.

### Architecture semantic drift

Risk: shared experts, scales, biases, LoRA, or pre-FFN weights make the owner path
numerically different.

Mitigation: gate the feature on one explicitly supported architecture and fail
closed for other architectures.

### Hidden transport fallback

Risk: an API described as peer copy may traverse host memory or synchronize in
an unexpected place.

Mitigation: provide explicit direct and explicit host-staged modes, record the
selected mode, and measure both copy legs independently.

## 17. Acceptance criteria

The milestone is complete only when all conditions below are demonstrated:

1. One configured 32-expert layer uses a layer-indexed 16/16 mapping.
2. GPU0 contains only its 16 packed experts for that routed layer after
   initialization, and GPU1 contains only its 16 packed experts.
3. No routed expert weight bytes move during measured generation.
4. Decode uses one activation transfer per selected remote owner, not per
   selected remote expert.
5. Each remote owner applies routing weights, reduces locally, and returns one
   partial hidden vector per token.
6. Target-layer output matches the unmodified path within a stated tolerance.
7. Final logits match within a stated tolerance and deterministic decode does
   not diverge for the acceptance prompt.
8. Measured logical payload equals
   `sum_t u_t*(H*b_in + R*b_out)`.
9. Routing metadata bytes are reported separately.
10. P2P, D2H, and H2D requested bytes and timings are reported separately.
11. Router-ID synchronization, owner GEMM, return, and combine timing are
    individually visible.
12. The trace proves ordering:

```text
prefix/router
  -> ID synchronization
  -> activation dispatch
  -> owner GEMMs
  -> owner reduction
  -> partial return
  -> origin combine
  -> suffix
```

13. Local-only, remote-only, mixed, and zero-owner-assignment cases pass.
14. Unsupported architectures and configurations fail with actionable errors.
15. A run with no static-placement parameters constructs no sidecar, allocates
    no owner buffers, and follows the existing MoE path.
16. A run with static placement enabled is the only run that creates owner
    tensors, dispatch staging, or prefix/suffix interception.
17. Requesting the reserved fixed-GPU mode fails as unsupported until its
    separate design and implementation are complete; it never silently runs the
    host mode.

> REMARK: PROPOSED ADDITION. These final two criteria are merge-safety gates.
> They should be tested before rebasing the feature onto a new llama.cpp main
> branch.

## 18. Final proposed flow

```text
Current:
  MoE layer
    -> all expert tensors follow whole-layer placement
    -> global IDs select slices in one complete tensor
    -> one backend executes each MUL_MAT_ID node

Prototype:
  target layer prefix on GPU0
    -> router produces layer-local global IDs and weights
    -> host reads IDs and indexes expert_to_owner[layer][expert]
    -> global IDs map through expert_to_local[layer][expert]
    -> one GPU-resident activation copy per active remote owner
    -> fixed-shape owner-local MoE graph uses stationary packed weights
    -> owner applies weights and reduces selected local experts
    -> one partial result returns per remote owner
    -> GPU0 combines owner partials
    -> target layer suffix continues under normal layer placement
```

This design intentionally accepts host synchronization and padded owner work in
exchange for a small, testable implementation boundary. The first milestone is
intended to answer whether stationary expert placement is correct and worthwhile
on the target PCIe x1 topology before changing generic ggml graph semantics.

## 19. Merge checklist for upstream llama.cpp updates

> REMARK: PROPOSED ADDITION. Use this checklist whenever rebasing the prototype
> onto a newer llama.cpp revision.

1. Confirm the default parameter initializer still sets static placement to
   disabled.
2. Confirm the normal `create_tensor_gate_up_exps()` and architecture-specific
   down-expert creation paths are unchanged when the flag is false.
3. Confirm the normal GGUF loader still owns all ordinary expert tensors when
   the flag is false.
4. Confirm the ordinary `build_moe_ffn()` path is selected when no sidecar exists.
5. Re-check the `MUL_MAT_ID` tensor contract and CUDA supported types.
6. Re-check graph reuse and warmup behavior.
7. Re-run the no-parameters regression test before running static-placement tests.
8. Re-run the synthetic two-backend correctness test and compare traffic
   counters with the previous revision.
9. Review all `REMARK: PROPOSED ADDITION` blocks against upstream changes before
   resolving conflicts.
