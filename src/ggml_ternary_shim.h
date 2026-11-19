#pragma once
#include <stddef.h>
#include <stdint.h>
#include "ggml.h"

// Experimental CPU F32 reference only. No CUDA, CTNN, GGUF-map loader,
// maintainer masks, or physical KV trimming are implemented.
struct ggml_ternary_parameters {
    const int8_t * codes;
    size_t code_stride;
    size_t code_count;
    bool is_h0_maintainer_slice;
};
enum ggml_ternary_status { GGML_TERNARY_OK = 0, GGML_TERNARY_INVALID = -1, GGML_TERNARY_UNSUPPORTED = -2 };
extern "C" int ggml_ternary_compute_forward(struct ggml_tensor * dst,
    const struct ggml_tensor * a, const struct ggml_tensor * b, const void * user_data);
extern "C" int ggml_ternary_load_map(const char * path, void ** out_codes_buffer,
    size_t * out_size, int * out_h0_maintainer_count);
