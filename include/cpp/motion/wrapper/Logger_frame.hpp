/*!
 * \file
 * \brief C++ wrapper to log 2D array of labels (after CCL).
 */

#pragma once

#include <stdint.h>
#include <aff3ct-core.hpp>

#include "motion/image/image_struct.h"
#include "motion/video/video_struct.h"

class Logger_frame : public aff3ct::module::Module {
protected:
    const std::string frames_path;
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    const uint8_t show_id;
    const uint32_t** in_labels;
    img_data_t* img_data;
    video_writer_t* video_writer;
public:
    Logger_frame(const std::string frames_path, const size_t fra_start, const int show_id, const int i0, const int i1,
                 const int j0, const int j1, const size_t max_RoIs_size);
    virtual ~Logger_frame();
};
