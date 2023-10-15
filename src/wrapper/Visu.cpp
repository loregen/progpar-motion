#include "motion/visu/visu_io.h"
#include "motion/tools.h"

#include "motion/wrapper/Visu.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Visu::Visu(const char* path, const size_t start, const size_t n_ffmpeg_threads, const int i0, const int i1,
           const int j0, const int j1, const enum pixfmt_e pixfmt, const enum video_codec_e codec_type,
           const uint8_t draw_track_id, const int win_play, const size_t buff_size, const size_t max_RoIs_size,
           const uint8_t skip_fra, const tracking_data_t* tracking_data)
: Module(), i0(i0), i1(i1), j0(j0), j1(j1), in_img(nullptr), tracking_data(tracking_data), visu_data(nullptr) {
    assert(tracking_data != NULL);

    const std::string name = "Visu";
    this->set_name(name);
    this->set_short_name(name);

    this->in_img = (const uint8_t**)malloc((size_t)(((i1 - i0) + 1) * sizeof(const uint8_t*)));
    this->in_img -= i0;

    this->visu_data = visu_alloc_init(path, start, n_ffmpeg_threads, (i1 - i0) + 1, (j1 - j0) + 1, pixfmt, codec_type,
                                      draw_track_id, win_play, buff_size, max_RoIs_size, skip_fra);

    auto socket_img_size = ((i1 - i0) + 1) * ((j1 - j0) + 1);

    auto &p = this->create_task("display");
    auto si_frame = this->template create_socket_in<uint32_t>(p, "in_frame", 1);
    auto si_img = this->template create_socket_in<uint8_t>(p, "in_img", socket_img_size);

    auto si_RoIs = this->template create_socket_in<uint8_t>(p, "in_RoIs", max_RoIs_size * sizeof(RoI_t));
    auto si_n_RoIs = this->template create_socket_in<uint32_t>(p, "in_n_RoIs", 1);
    // this is a hack to lauch this task after the tracking (visu[in_fake] = tracking[status])
    /* auto ps_in_fake = */ this->template create_socket_in<int32_t>(p, "in_fake", 1);

    this->create_codelet(p, [si_frame, si_img, si_RoIs, si_n_RoIs]
                            (Module &m, runtime::Task &t, const size_t frame_id) -> int {
        auto &vis = static_cast<Visu&>(m);

        const uint32_t frame = *t[si_frame].get_dataptr<const uint32_t>();

        tools_linear_2d_nrc_ui8matrix(t[si_img].get_dataptr<const uint8_t>(),
                                      vis.i0, vis.i1, vis.j0, vis.j1,
                                      vis.in_img);

        visu_display(vis.visu_data, vis.in_img,
                     t[si_RoIs].get_dataptr<const RoI_t>(),
                     *t[si_n_RoIs].get_dataptr<uint32_t>(),
                     vis.tracking_data->tracks,
                     frame);


        return aff3ct::runtime::status_t::SUCCESS;
    });
}

Visu::~Visu() {
    visu_free(this->visu_data);
    free(this->in_img + this->i0);
}

void Visu::flush() {
    visu_flush(this->visu_data, this->tracking_data->tracks);
}
