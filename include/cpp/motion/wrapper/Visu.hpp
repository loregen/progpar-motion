/*!
 * \file
 * \brief C++ wrapper for visualization.
 */

#pragma once

#include <stdint.h>
#include <streampu.hpp>

#include "motion/visu/visu_struct.h"

class Visu : public spu::module::Stateful {
protected:
    const int i0;
    const int i1;
    const int j0;
    const int j1;
    const tracking_data_t* tracking_data;
    visu_data_t* visu_data;
public:
    Visu(const char* path, const size_t start, const size_t n_ffmpeg_threads, const int i0, const int i1,
         const int j0, const int j1, const enum pixfmt_e pixfmt_in, const enum pixfmt_e pixfmt_out,
         const enum video_codec_e codec_type, const uint8_t draw_track_id, const int win_play, const bool ffmpeg_debug,
         const char* ffmpeg_out_extra_opts, const size_t buff_size, const size_t max_RoIs_size, const uint8_t skip_fra,
         const tracking_data_t* tracking_data);
    Visu(const char* path, const size_t start, const size_t n_ffmpeg_threads, const int i0, const int i1,
         const int j0, const int j1, const enum pixfmt_e pixfmt_in, const enum pixfmt_e pixfmt_out,
         const enum video_codec_e codec_type, const uint8_t draw_track_id, const int win_play, const size_t buff_size,
         const size_t max_RoIs_size, const uint8_t skip_fra, const tracking_data_t* tracking_data);
    virtual ~Visu();
    void flush();
};
