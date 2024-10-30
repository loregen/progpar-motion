/* Module with StreamPU compute task that calls features_extract*/

#include "motion/features/features_compute.h"

#include "motion/wrapper/Features_CCA.hpp"

CCA::CCA(const int i0, const int i1, const int j0, const int j1, const size_t max_size)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), max_size(max_size)
{
    const std::string name = "CCA";
    this->set_name(name);
    this->set_short_name(name);

    auto &c = this->create_task("extract");
    auto si_labels = this->template create_2d_socket_in<uint32_t>(c, "in_labels", ((i1 - i0) + 1), ((j1 - j0) + 1));
    auto so_roi = this->template create_socket_out<uint8_t>(c, "out_RoIs", max_size * sizeof(RoI_t));
    auto si_n_RoIs = this->template create_socket_in<uint32_t>(c, "in_n_RoIs", 1);

    this->create_codelet(c, [si_labels, so_roi, si_n_RoIs]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &cca = static_cast<CCA&>(m);
        
        const uint32_t** in_labels = t[si_labels].get_2d_dataptr<const uint32_t>();
        RoI_t* RoIs = t[so_roi].get_dataptr<RoI_t>();
        const uint32_t* n_RoIs = t[si_n_RoIs].get_dataptr<const uint32_t>();

        features_extract(in_labels, cca.i0, cca.i1, cca.j0, cca.j1, RoIs, *n_RoIs);

        return spu::runtime::status_t::SUCCESS;
    });
}

CCA::~CCA() {
    // CCA_LSL_free_data(this->cca_data);
}