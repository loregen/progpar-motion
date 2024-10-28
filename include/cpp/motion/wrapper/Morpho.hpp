/*!
 * \file
 * \brief C++ wrapper for morphology algorithms.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

#include "motion/morpho/morpho_struct.h"

class Morpho : public spu::module::Stateful {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    morpho_data_t* morpho_data;
public:
    Morpho(const int i0, const int i1, const int j0, const int j1, morpho_data_t* morpho_data);
    virtual ~Morpho();
};
