/*!
 * \file
 * \brief C++ wrapper to log RoIs features.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

#include "motion/tracking/tracking_struct.h"

class Logger_RoIs : public spu::module::Stateful {
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
