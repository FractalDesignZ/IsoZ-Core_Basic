#pragma once
#include <stddef.h>
#include <stdint.h>
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1 };
struct ggml_tensor { ggml_type type; int64_t ne[4]; size_t nb[4]; void * data; };
