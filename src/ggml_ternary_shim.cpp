/* Experimental CPU F32 reference for a custom ggml integration.
 * Build against the target fork's ggml.h. This is not a registered ggml op.
 * Codes route zero or original signed weights; code 1 is not implicitly +1.
 * No model-fidelity, homology, memory-bound, or accelerator claims are made.
 */
#include "ggml_ternary_shim.h"
#include <cstring>
#include <limits>

static bool supported(const ggml_tensor * tensor) {
    return tensor && tensor->data && tensor->type == GGML_TYPE_F32 &&
           tensor->ne[0] > 0 && tensor->ne[1] > 0 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
           tensor->nb[0] == sizeof(float) &&
           static_cast<size_t>(tensor->ne[0]) <= std::numeric_limits<size_t>::max() / sizeof(float) &&
           tensor->nb[1] >= static_cast<size_t>(tensor->ne[0]) * sizeof(float);
}

static float read_f32(const ggml_tensor * tensor, int64_t x, int64_t y) {
    float value;
    std::memcpy(&value, static_cast<const char *>(tensor->data) +
                static_cast<size_t>(x) * tensor->nb[0] + static_cast<size_t>(y) * tensor->nb[1], sizeof(value));
    return value;
}

extern "C" int ggml_ternary_compute_forward(ggml_tensor * dst,
    const ggml_tensor * a, const ggml_tensor * b, const void * user_data) {
    if (!supported(dst) || !supported(a) || !supported(b) || !user_data)
        return GGML_TERNARY_INVALID;
    // ggml dimensions: a=[K,M], b=[K,N], dst=[M,N].
    if (a->ne[0] != b->ne[0] || dst->ne[0] != a->ne[1] || dst->ne[1] != b->ne[1] ||
        dst->data == a->data || dst->data == b->data)
        return GGML_TERNARY_INVALID;
    const auto & params = *static_cast<const ggml_ternary_parameters *>(user_data);
    const size_t rows = static_cast<size_t>(a->ne[1]);
    const size_t inner = static_cast<size_t>(a->ne[0]);
    if (!params.codes || params.code_stride < inner ||
        rows > std::numeric_limits<size_t>::max() / params.code_stride ||
        params.code_count < rows * params.code_stride)
        return GGML_TERNARY_INVALID;
    // Validate the complete code buffer before writing any destination element.
    for (size_t row = 0; row < rows; ++row)
        for (size_t k = 0; k < inner; ++k) {
            const int8_t code = params.codes[row * params.code_stride + k];
            if (code != 0 && code != 1 && code != 3) return GGML_TERNARY_INVALID;
        }
    for (int64_t batch = 0; batch < b->ne[1]; ++batch)
        for (int64_t row = 0; row < a->ne[1]; ++row) {
            float sum = 0.0f;
            for (int64_t k = 0; k < a->ne[0]; ++k) {
                const int8_t code = params.codes[static_cast<size_t>(row) * params.code_stride + static_cast<size_t>(k)];
                if (code == 0 && !params.is_h0_maintainer_slice) continue;
                sum += read_f32(a, k, row) * read_f32(b, k, batch);
            }
            std::memcpy(static_cast<char *>(dst->data) + static_cast<size_t>(row) * dst->nb[0] +
                        static_cast<size_t>(batch) * dst->nb[1], &sum, sizeof(sum));
        }
    return GGML_TERNARY_OK;
}

extern "C" int ggml_ternary_load_map(const char * path, void ** out_codes_buffer,
    size_t * out_size, int * out_h0_maintainer_count) {
    if (out_codes_buffer) *out_codes_buffer = nullptr;
    if (out_size) *out_size = 0;
    if (out_h0_maintainer_count) *out_h0_maintainer_count = 0;
    if (!path || !out_codes_buffer || !out_size || !out_h0_maintainer_count)
        return GGML_TERNARY_INVALID;
    // An unimplemented loader must never return a success receipt.
    return GGML_TERNARY_UNSUPPORTED;
}
