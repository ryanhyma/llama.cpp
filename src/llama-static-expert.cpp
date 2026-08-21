#include "llama-static-expert.h"

#include <stdexcept>
#include <utility>

std::unique_ptr<llama_static_expert_placement> llama_static_expert_placement::create(
        const llama_static_expert_params * params,
        int32_t n_layer,
        int32_t n_expert) {
    if (params == nullptr || !params->enabled) {
        return nullptr;
    }

    if (params->dispatch_mode != LLAMA_STATIC_EXPERT_DISPATCH_HOST) {
        throw std::invalid_argument(
            "static expert dispatch mode is not implemented; use host dispatch");
    }
    if (n_layer <= 0 || n_expert <= 0) {
        throw std::invalid_argument(
            "static expert placement requires a model with MoE layers and experts");
    }
    if (params->target_layer < 0 || params->target_layer >= n_layer) {
        throw std::invalid_argument("static expert target_layer is out of range");
    }
    if (params->owner_count <= 0 || params->owner_count > n_expert) {
        throw std::invalid_argument("static expert owner_count is invalid");
    }
    if (params->expert_to_owner == nullptr ||
            params->expert_to_owner_count != n_expert) {
        throw std::invalid_argument(
            "static expert mapping must contain one owner for every expert");
    }

    std::vector<int32_t> expert_to_owner(n_expert);
    std::vector<int32_t> expert_to_local(n_expert, -1);
    std::vector<int32_t> local_counts(params->owner_count, 0);

    for (int32_t expert = 0; expert < n_expert; ++expert) {
        const int32_t owner = params->expert_to_owner[expert];
        if (owner < 0 || owner >= params->owner_count) {
            throw std::invalid_argument("static expert owner ID is out of range");
        }

        expert_to_owner[expert] = owner;
        expert_to_local[expert] = local_counts[owner]++;
    }

    for (int32_t owner = 0; owner < params->owner_count; ++owner) {
        if (local_counts[owner] == 0) {
            throw std::invalid_argument("static expert owner has no assigned experts");
        }
    }

    return std::unique_ptr<llama_static_expert_placement>(
        new llama_static_expert_placement(
            params->target_layer,
            params->owner_count,
            n_expert,
            std::move(expert_to_owner),
            std::move(expert_to_local),
            std::move(local_counts)));
}

llama_static_expert_placement::llama_static_expert_placement(
        int32_t target_layer,
        int32_t owner_count,
        int32_t expert_count,
        std::vector<int32_t> expert_to_owner,
        std::vector<int32_t> expert_to_local,
        std::vector<int32_t> local_counts)
    : enabled_(true),
      target_layer_(target_layer),
      owner_count_(owner_count),
      expert_count_(expert_count),
      expert_to_owner_(std::move(expert_to_owner)),
      expert_to_local_(std::move(expert_to_local)),
      local_counts_(std::move(local_counts)) {
}

int32_t llama_static_expert_placement::owner_for(int32_t global_expert) const {
    if (global_expert < 0 || global_expert >= expert_count_) {
        throw std::out_of_range("global expert ID is out of range");
    }
    return expert_to_owner_[global_expert];
}

int32_t llama_static_expert_placement::local_for(int32_t global_expert) const {
    if (global_expert < 0 || global_expert >= expert_count_) {
        throw std::out_of_range("global expert ID is out of range");
    }
    return expert_to_local_[global_expert];
}

int32_t llama_static_expert_placement::local_count(int32_t owner) const {
    if (owner < 0 || owner >= owner_count_) {
        throw std::out_of_range("expert owner ID is out of range");
    }
    return local_counts_[owner];
}
