/*!
 * \file
 * \brief C++ wrapper to log tracks.
 */

#pragma once

#include <stdint.h>
#include <aff3ct-core.hpp>

#include "motion/tracking/tracking_struct.h"

class Logger_tracks : public aff3ct::module::Module {
protected:
    const std::string tracks_path;
    const size_t fra_start;
    const tracking_data_t* tracking_data;
public:
    Logger_tracks(const std::string tracks_path, const size_t fra_start, const tracking_data_t* tracking_data);
    virtual ~Logger_tracks();
};

