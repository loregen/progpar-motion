/*!
 * \file
 * \brief C++ wrapper to log RoIs features.
 */

#pragma once

#include <stdint.h>
#include <aff3ct-core.hpp>

#include "motion/tracking/tracking_struct.h"

class Logger_RoIs : public aff3ct::module::Module {
protected:
    const std::string RoIs_path;
    const size_t fra_start;
    const size_t fra_skip;
    const tracking_data_t* tracking_data;
public:
    Logger_RoIs(const std::string RoIs_path, const size_t fra_start, const size_t frame_skip,
                const size_t max_RoIs_size, const tracking_data_t* tracking_data);
    virtual ~Logger_RoIs();
};
