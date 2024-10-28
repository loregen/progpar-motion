/* Module with StreamPU compute task that calls CCL_LSL_apply*/

#include "motion/CCL/CCL_compute.h"

#include "motion/wrapper/CCL.hpp"

CCL::CCL(const int i0, const int i1, const int j0, const int j1, CCL_data_t* ccl_data, uint8_t no_init_labels)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), ccl_data(ccl_data), no_init_labels(no_init_labels)
{
    assert(ccl_data != NULL);

    const std::string name = "CCL";
    this->set_name(name);
    this->set_short_name(name);

    auto &c = this->create_task("apply");
    auto si_img = this->template create_2d_socket_in<uint8_t>(c, "in_img", ((i1 - i0) + 1), ((j1 - j0) + 1));
    auto roi = this->template create_socket_out<uint32_t>(c, "out_RoIs", 1);
    auto lo_labels = this->template create_2d_socket_out<uint32_t>(c, "out_labels", ((i1 - i0) + 1), ((j1 - j0) + 1));

    this->create_codelet(c, [si_img, lo_labels, roi]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &ccl = static_cast<CCL&>(m);

        const uint8_t** in_img = t[si_img].get_2d_dataptr<const uint8_t>();
        uint32_t** out_label = t[lo_labels].get_2d_dataptr<uint32_t>();
        uint32_t* out_roi = t[roi].get_dataptr<uint32_t>();

        *out_roi = CCL_LSL_apply(ccl.ccl_data, in_img, out_label, ccl.no_init_labels);

        return spu::runtime::status_t::SUCCESS;
    });
}

CCL::~CCL() {
    CCL_LSL_free_data(this->ccl_data);
}