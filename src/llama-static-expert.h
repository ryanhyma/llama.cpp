#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <vector>

// Opt-in static expert metadata. This sidecar intentionally does not alter
// llama_layer's existing expert tensors or GGML_OP_MUL_MAT_ID semantics.
class llama_static_expert_placement {
public:
    static std::unique_ptr<llama_static_expert_placement> create(
            const llama_static_expert_params * params,
            int32_t n_layer,
            int32_t n_expert);

    bool enabled() const { return enabled_; }
    int32_t target_layer() const { return target_layer_; }
    int32_t owner_count() const { return owner_count_; }
    int32_t expert_count() const { return expert_count_; }

    int32_t owner_for(int32_t global_expert) const;
    int32_t local_for(int32_t global_expert) const;
    int32_t local_count(int32_t owner) const;

    const std::vector<int32_t> & expert_to_owner() const { return expert_to_owner_; }
    const std::vector<int32_t> & expert_to_local() const { return expert_to_local_; }

private:
    llama_static_expert_placement(
            int32_t target_layer,
            int32_t owner_count,
            int32_t expert_count,
            std::vector<int32_t> expert_to_owner,
            std::vector<int32_t> expert_to_local,
            std::vector<int32_t> local_counts);

    bool enabled_ = false;
    int32_t target_layer_ = -1;
    int32_t owner_count_ = 0;
    int32_t expert_count_ = 0;

    std::vector<int32_t> expert_to_owner_;
    std::vector<int32_t> expert_to_local_;
    std::vector<int32_t> local_counts_;
};
