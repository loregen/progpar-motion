#include <nrc2.h>

#include "motion/tools.h"
#include "motion/image/image_compute.h"
#include "motion/video/video_io.h"

#include "motion/wrapper/Logger_frame.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Logger_frame::Logger_frame(const std::string frames_path, const size_t fra_start, const int show_id, const int i0,
                           const int i1, const int j0, const int j1, const size_t max_RoIs_size)
: Module(), i0(i0), i1(i1), j0(j0), j1(j1), show_id(show_id), in_labels(nullptr), img_data(nullptr),
  video_writer(nullptr) {
    const std::string name = "Logger_frame";
    this->set_name(name);
    this->set_short_name(name);

    this->in_labels = (const uint32_t**)malloc((size_t)(((i1 - i0) + 1) * sizeof(const uint32_t*)));
    this->in_labels -= i0;

    this->img_data = image_gs_alloc((i1 - i0) + 1, (j1 - j0) + 1);
    const size_t n_threads = 1;
    this->video_writer = video_writer_alloc_init(frames_path.c_str(), fra_start, n_threads, (i1 - i0) + 1,
                                                 (j1 - j0) + 1, PIXFMT_GRAY, VCDC_FFMPEG_IO, 0);

    auto socket_img_size = ((i1 - i0) + 1) * ((j1 - j0) + 1);
    auto &t = this->create_task("write");
    auto si_labels = this->template create_socket_in<uint32_t>(t, "in_labels", socket_img_size);
    auto si_RoIs = this->template create_socket_in<uint8_t>(t, "in_RoIs", max_RoIs_size * sizeof(RoI_t));
    auto si_n_RoIs = this->template create_socket_in<uint32_t>(t, "in_n_RoIs", 1);

    this->create_codelet(t, [si_labels, si_RoIs, si_n_RoIs]
                            (Module &m, runtime::Task &t, const size_t frame_id) -> int {
        auto &lgr_fra = static_cast<Logger_frame&>(m);

        tools_linear_2d_nrc_ui32matrix(t[si_labels].get_dataptr<const uint32_t>(),
                                       lgr_fra.i0, lgr_fra.i1, lgr_fra.j0, lgr_fra.j1,
                                       lgr_fra.in_labels);

        image_gs_draw_labels(lgr_fra.img_data, lgr_fra.in_labels,
                             t[si_RoIs].get_dataptr<const RoI_t>(),
                             *t[si_n_RoIs].get_dataptr<uint32_t>(),
                             lgr_fra.show_id);


        video_writer_save_frame(lgr_fra.video_writer, (const uint8_t**)image_gs_get_pixels_2d(lgr_fra.img_data));

        return aff3ct::runtime::status_t::SUCCESS;
    });
}

Logger_frame::~Logger_frame() {
    free(this->in_labels + this->i0);
    image_gs_free(this->img_data);
    video_writer_free(this->video_writer);
}
