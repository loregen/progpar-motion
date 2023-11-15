/*!
 * \file
 * \brief C++ wrapper for visualization.
 */

#pragma once

#include <stdint.h>
#include <aff3ct-core.hpp>

#include "motion/visu/visu_struct.h"

class Visu : public aff3ct::module::Module {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    const tracking_data_t* tracking_data;
    visu_data_t* visu_data;
public:
    Visu(const char* path, const size_t start, const size_t n_ffmpeg_threads, const int i0, const int i1,
         const int j0, const int j1, const enum pixfmt_e pixfmt, const enum video_codec_e codec_type,
         const uint8_t draw_track_id, const int win_play, const size_t buff_size, const size_t max_RoIs_size,
         const uint8_t skip_fra, const tracking_data_t* tracking_data);
    virtual ~Visu();
    void flush();
};
