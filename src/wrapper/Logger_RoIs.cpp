#include "motion/features/features_io.h"
#include "motion/tools.h"

#include "motion/wrapper/Logger_RoIs.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Logger_RoIs::Logger_RoIs(const std::string RoIs_path, const size_t fra_start, const size_t fra_skip,
                         const size_t max_RoIs_size, const tracking_data_t* tracking_data)
: Module(), RoIs_path(RoIs_path), fra_start(fra_start), fra_skip(fra_skip), tracking_data(tracking_data) {
    assert(tracking_data != NULL);

    const std::string name = "Logger_RoIs";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("write");
    auto si_RoIs0 = this->template create_socket_in<uint8_t>(t, "in_RoIs0", max_RoIs_size * sizeof(RoI_t));
    auto si_n_RoIs0 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs0", 1);
    auto si_RoIs1 = this->template create_socket_in<uint8_t>(t, "in_RoIs1", max_RoIs_size * sizeof(RoI_t));
    auto si_n_RoIs1 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs1", 1);
    auto si_frame = this->template create_socket_in<uint32_t>(t, "in_frame", 1);

    if (!RoIs_path.empty())
        tools_create_folder(RoIs_path.c_str());

    this->create_codelet(t, [si_RoIs0, si_n_RoIs0, si_RoIs1, si_n_RoIs1, si_frame]
                            (Module &m, runtime::Task &t, const size_t frame_id) -> int {
        auto &lgr_roi = static_cast<Logger_RoIs&>(m);

        const uint32_t frame = *t[si_frame].get_dataptr<const uint32_t>();
        if (!lgr_roi.RoIs_path.empty()) {
            char file_path[256];
            snprintf(file_path, sizeof(file_path), "%s/%05u.txt", lgr_roi.RoIs_path.c_str(), frame);
            FILE* file = fopen(file_path, "w");
            int prev_frame = frame > lgr_roi.fra_start ? (int)frame - (lgr_roi.fra_skip + 1) : -1;
            features_RoIs0_RoIs1_write(file, prev_frame, frame,
                                       t[si_RoIs0].get_dataptr<const RoI_t>(),
                                       *t[si_n_RoIs0].get_dataptr<uint32_t>(),
                                       t[si_RoIs1].get_dataptr<const RoI_t>(),
                                       *t[si_n_RoIs1].get_dataptr<uint32_t>(),
                                       lgr_roi.tracking_data->tracks);
            fclose(file);
        }
        return aff3ct::runtime::status_t::SUCCESS;
    });
}

Logger_RoIs::~Logger_RoIs() {}
