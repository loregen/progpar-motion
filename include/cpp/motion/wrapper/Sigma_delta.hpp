/*!
 * \file
 * \brief C++ wrapper for sigma_delta algorithm.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

#include "motion/sigma_delta/sigma_delta_struct.h"

class Sigma_delta : public spu::module::Stateful {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    const uint8_t N;
    sigma_delta_data_t* sd_data;
public:
    Sigma_delta(const int i0, const int i1, const int j0, const int j1, const uint8_t vmin, const uint8_t vmax, const uint8_t N);
    virtual ~Sigma_delta();
};
