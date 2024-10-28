/* Module with StreamPU compute task that calls morpho_compute_opening3 and morpho_compute_closing3*/

#include "motion/morpho/morpho_compute.h"

#include "motion/wrapper/Morpho.hpp"

Morpho::Morpho(const int i0, const int i1, const int j0, const int j1, morpho_data_t* morpho_data)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), morpho_data(morpho_data)
{
    const std::string name = "Morpho";
    this->set_name(name);
    this->set_short_name(name);

    auto &c = this->create_task("compute");
    auto si_img = this->template create_2d_socket_in<uint8_t>(c, "in_img", ((i1 - i0) + 1), ((j1 - j0) + 1));

    auto so_img = this->template create_2d_socket_out<uint8_t>(c, "out_img", ((i1 - i0) + 1), ((j1 - j0) + 1));

    this->create_codelet(c, [si_img, so_img]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &mrp = static_cast<Morpho&>(m);

        const uint8_t** in_img = t[si_img].get_2d_dataptr<const uint8_t>();
        uint8_t** out_img = t[so_img].get_2d_dataptr<uint8_t>();

        morpho_compute_opening3(mrp.morpho_data, in_img, out_img, mrp.i0, mrp.i1, mrp.j0, mrp.j1);
        morpho_compute_closing3(mrp.morpho_data, in_img, out_img, mrp.i0, mrp.i1, mrp.j0, mrp.j1);

        return spu::runtime::status_t::SUCCESS;
    });
}

Morpho::~Morpho() {
    morpho_free_data(this->morpho_data);
}