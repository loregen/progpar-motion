/*!
 * \file
 * \brief C++ wrapper to get image at \f$t\f$.
 */

#pragma once

#include <stdint.h>
#include <aff3ct-core.hpp>

#include "motion/video/video_struct.h"

class Video_reader : public aff3ct::module::Module,
                     public aff3ct::tools::Interface_is_done
{
protected:
    int i0, i1, j0, j1;
    video_reader_t* video;
    bool done;
public:
    Video_reader(const std::string filename, const size_t frame_start, const size_t frame_end, const size_t frame_skip,
                 const int bufferize, const size_t n_ffmpeg_threads,
                 const enum video_codec_e codec_type = VCDC_FFMPEG_IO,
                 const enum video_codec_hwaccel_e hwaccel = VCDC_HWACCEL_NONE);
    virtual ~Video_reader();

    virtual bool is_done() const;
    int get_i0();
    int get_i1();
    int get_j0();
    int get_j1();
    void set_loop_size(size_t loop_size);
};
