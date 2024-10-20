#include <cstdlib>
#include <cstring>

#include "memory-processing.h"

namespace memproc
{
    template<>
    bool dealloc<MemType::CPU>(void *&array, size_t &allocated_size)
    {
        if(allocated_size > 0)
        {
            free(array);
            allocated_size = 0;
        }

        return true;
    }

    template<>
    bool dealloc<MemType::CPU>(void *&array)
    {
        free(array);
        return true;
    }

    template <>
    bool alloc<MemType::CPU>(void *&array, const size_t new_size)
    {
        array = malloc(new_size);
        memset(array, 0, new_size);

        return true;
    }


    template <>
    bool realloc<MemType::CPU>(void *&array, size_t &allocated_size, const size_t new_size)
    {
        if(new_size > allocated_size)
        {
            if(allocated_size > 0) dealloc<MemType::CPU>(array, allocated_size);
            allocated_size = new_size;
            array = malloc(new_size);
            memset(array, 0, new_size);
        }

        return true;
    }

    template <>
    bool memcopy<MemType::CPU, MemType::CPU>(void *dst, const void* src, const size_t copy_elem_size)
    {
        if(copy_elem_size == 0 || dst == src) return true;
        memcpy(dst, src, copy_elem_size);

        return true;
    }
}