#include <nrc2.h>

#include "motion/tools.h"
#include "motion/image/image_compute.h"
#include "motion/video/video_io.h"

#include "motion/wrapper/Logger_frame.hpp"

Logger_frame::Logger_frame(const std::string frames_path, const size_t fra_start, const int show_id, const int i0,
                           const int i1, const int j0, const int j1, const size_t max_RoIs_size)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), show_id(show_id), img_data(nullptr), video_writer(nullptr)
{
    const std::string name = "Logger_frame";
    this->set_name(name);
    this->set_short_name(name);

    this->img_data = image_gs_alloc((i1 - i0) + 1, (j1 - j0) + 1);
    const size_t n_threads = 1;
    this->video_writer = video_writer_alloc_init(frames_path.c_str(), fra_start, n_threads, (i1 - i0) + 1,
                                                 (j1 - j0) + 1, PIXFMT_GRAY8, VCDC_FFMPEG_IO, 0, 0, NULL);

    auto &t = this->create_task("write");
    auto si_labels = this->template create_2d_socket_in<uint32_t>(t, "in_labels", (i1 - i0) + 1, (j1 - j0) + 1);
    auto si_RoIs = this->template create_socket_in<uint8_t>(t, "in_RoIs", max_RoIs_size * sizeof(RoI_t));
    auto si_n_RoIs = this->template create_socket_in<uint32_t>(t, "in_n_RoIs", 1);

    this->create_codelet(t, [si_labels, si_RoIs, si_n_RoIs]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &lgr_fra = static_cast<Logger_frame&>(m);

        // calling get_2d_dataptr() has a small cost (it performs the 1D to 2D conversion)
        const uint32_t** in_labels = t[si_labels].get_2d_dataptr<const uint32_t>();

        image_gs_draw_labels(lgr_fra.img_data, in_labels,
                             t[si_RoIs].get_dataptr<const RoI_t>(),
                             *t[si_n_RoIs].get_dataptr<const uint32_t>(),
                             lgr_fra.show_id);


        video_writer_save_frame(lgr_fra.video_writer, (const uint8_t**)image_gs_get_pixels_2d(lgr_fra.img_data));

        return spu::runtime::status_t::SUCCESS;
    });
}

Logger_frame::~Logger_frame() {
    image_gs_free(this->img_data);
    video_writer_free(this->video_writer);
}
