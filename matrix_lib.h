#pragma once
#include "matrix.h"
#include <string>

namespace matrix_lib
{
    struct pyobj_info
    {
        void *ptr;
        int nrow, ncol;
        std::string format;
        MemType mem_type;
    };

    template <typename T, MemType dev_run>
    void mat_mul( pyobj_info& info_A, pyobj_info& info_B, T*& Res_ptr); 
}