/* Module with StreamPU compute task that calls CCL_LSL_apply*/

#include "motion/CCL/CCL_compute.h"

#include "motion/wrapper/CCL.hpp"

CCL::CCL(const int i0, const int i1, const int j0, const int j1, uint8_t no_init_labels)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), ccl_data(nullptr), no_init_labels(no_init_labels)
{

    const std::string name = "CCL";
    this->set_name(name);
    this->set_short_name(name);

    this->ccl_data = CCL_LSL_alloc_data(i0, i1, j0, j1);
    CCL_LSL_init_data(this->ccl_data);

    auto &c = this->create_task("apply");
    auto si_img = this->template create_2d_socket_in<uint8_t>(c, "in_img", ((i1 - i0) + 1), ((j1 - j0) + 1));
    auto roi = this->template create_socket_out<uint32_t>(c, "out_n_RoIs", 1);
    auto so_labels = this->template create_2d_socket_out<uint32_t>(c, "out_labels", ((i1 - i0) + 1), ((j1 - j0) + 1));

    this->create_codelet(c, [si_img, so_labels, roi]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &ccl = static_cast<CCL&>(m);

        const uint8_t** in_img = t[si_img].get_2d_dataptr<const uint8_t>();
        uint32_t** out_label = t[so_labels].get_2d_dataptr<uint32_t>();
        uint32_t* out_roi = t[roi].get_dataptr<uint32_t>();
        
        *out_roi = CCL_LSL_apply(ccl.ccl_data, in_img, out_label, ccl.no_init_labels);

        return spu::runtime::status_t::SUCCESS;
    });
}

CCL::~CCL() {
    CCL_LSL_free_data(this->ccl_data);
}