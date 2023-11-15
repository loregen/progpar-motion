/*!
 * \file
 * \brief C++ wrapper to log kNN statistics.
 */

#pragma once

#include <stdint.h>
#include <aff3ct-core.hpp>

class Logger_kNN : public aff3ct::module::Module {
protected:
    const std::string kNN_path;
    const size_t fra_start;
    const size_t max_size;
public:
    Logger_kNN(const std::string kNN_path, const size_t fra_start, const size_t max_size);
    virtual ~Logger_kNN();
};
