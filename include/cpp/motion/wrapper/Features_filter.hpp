/*!
 * \file
 * \brief C++ wrapper for dimensional filtering algorithm.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

class Features_filter : public spu::module::Stateful {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    size_t max_size_f;
    size_t min_size_f;
    size_t max_size_roi;
    size_t max_size_roi2;
public:
    Features_filter(const int i0, const int i1, const int j0, const int j1, const size_t max_size_f,
                     const size_t min_size_f, const size_t max_size_roi, const size_t max_size_roi2);
    virtual ~Features_filter();
};