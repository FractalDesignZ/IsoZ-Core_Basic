#include "ggml_ternary_shim.h"
#include <cassert>
#include <cstdint>

static ggml_tensor tensor(float * data, int64_t x, int64_t y) {
    return {GGML_TYPE_F32, {x,y,1,1}, {4,static_cast<size_t>(x)*4,0,0}, data};
}
int main() {
    float weights[] = {-2, 3, 4, -5, 6, 7};
    float inputs[] = {1,2,3,4,5,6};
    float output[] = {99,99,99,99};
    auto a=tensor(weights,3,2), b=tensor(inputs,3,2), dst=tensor(output,2,2);
    int8_t codes[] = {1,0,3,0,1,3};
    ggml_ternary_parameters params = {codes,3,6,false};
    assert(ggml_ternary_compute_forward(&dst,&a,&b,&params)==GGML_TERNARY_OK);
    assert(output[0]==10 && output[1]==33 && output[2]==16 && output[3]==72);
    params.is_h0_maintainer_slice = true;
    assert(ggml_ternary_compute_forward(&dst,&a,&b,&params)==GGML_TERNARY_OK);
    assert(output[0]==16 && output[1]==28 && output[2]==31 && output[3]==52);
    params.code_count=2;
    assert(ggml_ternary_compute_forward(&dst,&a,&b,&params)==GGML_TERNARY_INVALID);
    assert(output[0]==16);
    params.code_count=6; codes[0]=2;
    assert(ggml_ternary_compute_forward(&dst,&a,&b,&params)==GGML_TERNARY_INVALID);
    assert(ggml_ternary_compute_forward(nullptr,&a,&b,&params)==GGML_TERNARY_INVALID);
    void * buffer=reinterpret_cast<void *>(static_cast<uintptr_t>(1)); size_t size=99; int count=99;
    assert(ggml_ternary_load_map("synthetic.json",&buffer,&size,&count)==GGML_TERNARY_UNSUPPORTED);
    assert(buffer==nullptr && size==0 && count==0);
    assert(ggml_ternary_load_map(nullptr,nullptr,&size,&count)==GGML_TERNARY_INVALID);
}
