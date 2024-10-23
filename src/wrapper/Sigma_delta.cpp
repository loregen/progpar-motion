#include "motion/sigma_delta/sigma_delta_compute.h"

#include "motion/wrapper/Sigma_delta.hpp"

Sigma_delta::Sigma_delta(const int i0, const int i1, const int j0, const int j1, const uint8_t vmin, const uint8_t vmax, const uint8_t N)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), N(N)
{
    assert(sd_data != NULL);

    const std::string name = "Sigma_delta";
    this->set_name(name);
    this->set_short_name(name);

    this->sd_data = sigma_delta_alloc_data(i0, i1, j0, j1, vmin, vmax);

    auto &c = this->create_task("compute");
    auto si_img = this->template create_2d_socket_in<uint8_t>(c, "in_img", ((i1 - i0) + 1), ((j1 - j0) + 1));

    auto so_img = this->template create_2d_socket_out<uint8_t>(c, "out_img", ((i1 - i0) + 1), ((j1 - j0) + 1));

    //restart from here -------------------------------

    this->create_codelet(p, [si_frame, si_img, si_RoIs, si_n_RoIs]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &vis = static_cast<Visu&>(m);

        // calling get_2d_dataptr() has a small cost (it performs the 1D to 2D conversion)
        const uint8_t** in_img = t[si_img].get_2d_dataptr<const uint8_t>();

        visu_display(vis.visu_data,
                     in_img,
                     t[si_RoIs].get_dataptr<const RoI_t>(),
                     *t[si_n_RoIs].get_dataptr<const uint32_t>(),
                     vis.tracking_data->tracks,
                     *t[si_frame].get_dataptr<const uint32_t>());

        return spu::runtime::status_t::SUCCESS;
    });
}

Sigma_delta::~Sigma_delta() {
    sigma_delta_free_data(this->sd_data);
}
