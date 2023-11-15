#include "motion/video/video_io.h"
#include "motion/tools.h"

#include "motion/wrapper/Video_reader.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Video_reader::Video_reader(const std::string filename, const size_t frame_start, const size_t frame_end,
                           const size_t frame_skip, const int bufferize, const size_t n_ffmpeg_threads,
                           const enum video_codec_e codec_type, const enum video_codec_hwaccel_e hwaccel)
: Module(), i0(0), i1(0), j0(0), j1(0), video(nullptr), done(false) {
    const std::string name = "Video";
    this->set_name(name);
    this->set_short_name(name);

    this->video = video_reader_alloc_init(filename.c_str(), frame_start, frame_end, frame_skip, bufferize,
                                          n_ffmpeg_threads, codec_type, hwaccel, &this->i0, &this->i1, &this->j0,
                                          &this->j1);

    auto &t = this->create_task("generate");
    auto so_img = this->template create_2d_socket_out<uint8_t>(t, "out_img", (i1 - i0) + 1, (j1 - j0) + 1);
    auto so_frame = this->template create_socket_out<uint32_t>(t, "out_frame", 1);

    this->create_codelet(t, [so_img, so_frame](Module &m, runtime::Task &t,const size_t frame_id) -> int
    {
        auto &vid = static_cast<Video_reader&>(m);

        // calling get_2d_dataptr() has a small cost (it performs the 1D to 2D conversion)
        uint8_t** out_img = t[so_img].get_2d_dataptr<uint8_t>();

        int cur_fra = video_reader_get_frame(vid.video, out_img);
        vid.done = cur_fra == -1 ? true : false;
        if (vid.done)
            throw tools::processing_aborted(__FILE__, __LINE__, __func__);

        uint32_t* ptr_frame = t[so_frame].get_dataptr<uint32_t>();
        *ptr_frame = (uint32_t)cur_fra;

        return runtime::status_t::SUCCESS;
    });
}

Video_reader::~Video_reader() {
    video_reader_free(this->video);
}

bool Video_reader::is_done() const {
    return this->done;
}

int Video_reader::get_i0() {
    return this->i0;
}

int Video_reader::get_i1() {
    return this->i1;
}

int Video_reader::get_j0() {
    return this->j0;
}

int Video_reader::get_j1() {
    return this->j1;
}

void Video_reader::set_loop_size(size_t loop_size) {
    this->video->loop_size = loop_size;
}
