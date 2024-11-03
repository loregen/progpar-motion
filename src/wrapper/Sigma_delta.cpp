/*Module with StreamPU compute task that calls sigma_delta_compute function*/

#include "motion/sigma_delta/sigma_delta_compute.h"

#include "motion/wrapper/Sigma_delta.hpp"

Sigma_delta::Sigma_delta(const int i0, const int i1, const int j0, const int j1, const uint8_t N)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), N(N), sd_data(nullptr)
{
  
    const std::string name = "Sigma_delta";
    this->set_name(name);
    this->set_short_name(name);

    this->sd_data = sigma_delta_alloc_data(i0, i1, j0, j1, 1, 254);

    auto &c = this->create_task("compute");
    auto si_img = this->template create_2d_socket_in<uint8_t>(c, "in_img", ((i1 - i0) + 1), ((j1 - j0) + 1));

    auto so_img = this->template create_2d_socket_out<uint8_t>(c, "out_img", ((i1 - i0) + 1), ((j1 - j0) + 1));

    this->create_codelet(c, [si_img, so_img]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &sd = static_cast<Sigma_delta&>(m);

        const uint8_t** in_img = t[si_img].get_2d_dataptr<const uint8_t>();
        uint8_t** out_img = t[so_img].get_2d_dataptr<uint8_t>();

        sigma_delta_compute(sd.sd_data, in_img, out_img, sd.i0, sd.i1, sd.j0, sd.j1, sd.N);

        return spu::runtime::status_t::SUCCESS;
    });
}

void Sigma_delta::init_data(const uint8_t** in_img) {
    sigma_delta_init_data(this->sd_data, in_img, this->i0, this->i1, this->j0, this->j1);
}

Sigma_delta::~Sigma_delta() {
    sigma_delta_free_data(this->sd_data);
}