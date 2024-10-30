/* Module with StreamPU compute task that calls features_filter_surface
and features_shrink_basic functions*/

#include "motion/features/features_compute.h"

#include "motion/wrapper/Features_filter.hpp"

Features_filter::Features_filter(const int i0, const int i1, const int j0, const int j1, const size_t max_size_f, const size_t min_size_f, const size_t max_size_roi, const size_t max_size_roi2)
: spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), max_size_f(max_size_f), min_size_f(min_size_f), max_size_roi(max_size_roi), max_size_roi2(max_size_roi2)
{
    const std::string name = "Features_filter";
    this->set_name(name);
    this->set_short_name(name);

    auto &c = this->create_task("filter");
    auto si_labels = this->template create_2d_socket_in<uint32_t>(c, "in_labels", ((i1 - i0) + 1), ((j1 - j0) + 1));
    //auto so_labels = this->template create_2d_socket_out<uint32_t>(c, "out_labels", ((i1 - i0) + 1), ((j1 - j0) + 1));
    auto so_roi_tmp = this->template create_socket_out<uint8_t>(c, "out_RoIs_tmp", max_size_roi * sizeof(RoI_t));
    auto so_roi = this->template create_socket_out<uint8_t>(c, "out_RoIs", max_size_roi2 * sizeof(RoI_t));
    auto si_n_RoIs = this->template create_socket_in<uint32_t>(c, "in_n_RoIs", 1);
    auto so_n_RoIs = this->template create_socket_out<uint32_t>(c, "out_n_RoIs", 1);

    this->create_codelet(c, [si_labels, /*so_labels,*/ so_roi, so_roi_tmp, si_n_RoIs, max_size_roi2, so_n_RoIs]
                            (spu::module::Module &m, spu::runtime::Task &t, const size_t frame_id) -> int {
        auto &ff = static_cast<Features_filter&>(m);
        
        const uint32_t** in_labels = t[si_labels].get_2d_dataptr<const uint32_t>();
        //uint32_t** out_labels = t[so_labels].get_2d_dataptr<uint32_t>();
        RoI_t* RoIs_tmp = t[so_roi_tmp].get_dataptr<RoI_t>();
        RoI_t* RoIs = t[so_roi].get_dataptr<RoI_t>();
        const uint32_t* n_RoIs = t[si_n_RoIs].get_dataptr<const uint32_t>();
        uint32_t* n_RoIs_out = t[so_n_RoIs].get_dataptr<uint32_t>();

        *n_RoIs_out = features_filter_surface(in_labels, NULL, ff.i0, ff.i1, ff.j0, ff.j1, RoIs_tmp, *n_RoIs, ff.min_size_f, ff.max_size_f);
        assert(*n_RoIs_out <= (uint32_t)max_size_roi2);
        features_shrink_basic(RoIs_tmp, *n_RoIs, RoIs);

        return spu::runtime::status_t::SUCCESS;
    });
}

Features_filter::~Features_filter() {
    // CCA_LSL_free_data(this->cca_data);
}
