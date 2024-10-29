/*!
 * \file
 * \brief C++ wrapper for CCA features extraction algorithm.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

class CCA : public spu::module::Stateful {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    size_t max_size;
public:
    CCA(const int i0, const int i1, const int j0, const int j1, const size_t max_size);
    virtual ~CCA();
};