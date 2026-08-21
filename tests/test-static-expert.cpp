#include "llama-static-expert.h"

#include <stdexcept>

static void expect(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static llama_static_expert_params make_params(const int32_t * owners, int32_t count) {
    return llama_static_expert_params {
        /*.enabled                =*/ true,
        /*.target_layer           =*/ 3,
        /*.owner_count            =*/ 2,
        /*.expert_to_owner        =*/ owners,
        /*.expert_to_owner_count  =*/ count,
        /*.dispatch_mode          =*/ LLAMA_STATIC_EXPERT_DISPATCH_HOST,
        /*.require_peer_copy      =*/ false,
    };
}

int main() {
    // No parameters means no sidecar and is the compatibility/default path.
    expect(llama_static_expert_placement::create(nullptr, 8, 32) == nullptr,
        "disabled placement created a sidecar");

    const int32_t owners[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const auto params = make_params(owners, 32);
    const auto placement = llama_static_expert_placement::create(&params, 8, 32);

    expect(placement != nullptr, "valid placement was not created");
    expect(placement->enabled(), "valid placement is disabled");
    expect(placement->target_layer() == 3, "target layer was not retained");
    expect(placement->owner_count() == 2, "owner count was not retained");
    expect(placement->expert_count() == 32, "expert count was not retained");
    expect(placement->owner_for(0) == 0, "expert 0 owner mismatch");
    expect(placement->local_for(0) == 0, "expert 0 local ID mismatch");
    expect(placement->owner_for(15) == 0, "expert 15 owner mismatch");
    expect(placement->local_for(15) == 15, "expert 15 local ID mismatch");
    expect(placement->owner_for(16) == 1, "expert 16 owner mismatch");
    expect(placement->local_for(16) == 0, "expert 16 local ID mismatch");
    expect(placement->owner_for(31) == 1, "expert 31 owner mismatch");
    expect(placement->local_for(31) == 15, "expert 31 local ID mismatch");
    expect(placement->local_count(0) == 16, "owner 0 local count mismatch");
    expect(placement->local_count(1) == 16, "owner 1 local count mismatch");

    bool threw = false;
    try {
        auto invalid = params;
        invalid.target_layer = 8;
        llama_static_expert_placement::create(&invalid, 8, 32);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "out-of-range target layer was accepted");

    threw = false;
    try {
        auto invalid = params;
        invalid.dispatch_mode = LLAMA_STATIC_EXPERT_DISPATCH_FIXED_GPU;
        llama_static_expert_placement::create(&invalid, 8, 32);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "reserved dispatch mode was accepted");

    return 0;
}
