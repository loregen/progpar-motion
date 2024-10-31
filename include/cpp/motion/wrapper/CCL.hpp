/*!
 * \file
 * \brief C++ wrapper for connected components labelling algorithm.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

#include "motion/CCL/CCL_struct.h"

class CCL : public spu::module::Stateful {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    CCL_data_t* ccl_data;
    uint8_t no_init_labels;
public:
    CCL(const int i0, const int i1, const int j0, const int j1, uint8_t no_init_labels);
    virtual ~CCL();
};