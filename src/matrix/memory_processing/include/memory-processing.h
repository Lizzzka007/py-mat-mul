#pragma once
#include <cstddef>

enum class MemType
{
#ifdef ENABLE_GPU_COMPUTATIONS
    GPU,
#endif
    CPU
};

namespace memproc
{
    template <MemType memtype>
    bool alloc(void *&array, const size_t new_size);

    template <MemType memtype>
    bool realloc(void *&array, size_t &allocated_size, const size_t new_size);

    template<MemType memtype>
    bool dealloc(void *&array, size_t &allocated_size);

    template<MemType memtype>
    bool dealloc(void *&array);

    template <MemType dst_memtype, MemType src_memtype>
    bool memcopy(void *dst, const void* src, const size_t copy_elem_size);
}
