#include "motion/kNN/kNN_io.h"
#include "motion/tools.h"

#include "motion/wrapper/Logger_kNN.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Logger_kNN::Logger_kNN(const std::string kNN_path, const size_t fra_start, const size_t max_size)
: Module(), kNN_path(kNN_path), fra_start(fra_start), max_size(max_size) {
    const std::string name = "Logger_kNN";
    this->set_name(name);
    this->set_short_name(name);

    this->in_data_nearest = (const uint32_t**)malloc((size_t)(max_size * sizeof(uint32_t*)));
    this->in_data_distances = (const float**)malloc((size_t)(max_size * sizeof(float*)));

    auto &t = this->create_task("write");

    auto si_data_nearest = this->template create_socket_in<uint32_t>(t, "in_data_nearest", max_size * max_size);
    auto si_data_distances = this->template create_socket_in<float>(t, "in_data_distances", max_size * max_size);
#ifdef MOTION_ENABLE_DEBUG
    auto si_data_conflicts = this->template create_socket_in<uint32_t>(t, "in_data_conflicts", max_size);
#endif
    auto si_RoIs0 = this->template create_socket_in<uint8_t>(t, "in_RoIs0", max_size * sizeof(RoI_t));
    auto si_n_RoIs0 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs0", 1);
    auto si_RoIs1 = this->template create_socket_in<uint8_t>(t, "in_RoIs1", max_size * sizeof(RoI_t));
    auto si_n_RoIs1 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs1", 1);
    auto si_frame = this->template create_socket_in<uint32_t>(t, "in_frame", 1);

    if (!kNN_path.empty())
        tools_create_folder(kNN_path.c_str());

#ifdef MOTION_ENABLE_DEBUG
    this->create_codelet(t, [si_data_nearest, si_data_distances, si_data_conflicts, si_RoIs0, si_n_RoIs0, si_RoIs1,
                             si_n_RoIs1, si_frame]
#else
    this->create_codelet(t, [si_data_nearest, si_data_distances, si_RoIs0, si_n_RoIs0, si_RoIs1, si_n_RoIs1, si_frame]
#endif
                            (Module &m, runtime::Task &t, const size_t frame_id) -> int {
        auto &lgr_knn = static_cast<Logger_kNN&>(m);
        tools_linear_2d_nrc_ui32matrix(t[si_data_nearest].get_dataptr<const uint32_t>(), 0, lgr_knn.max_size -1, 0,
                                       lgr_knn.max_size -1, (const uint32_t**)lgr_knn.in_data_nearest);
        tools_linear_2d_nrc_f32matrix(t[si_data_distances].get_dataptr<const float>(), 0, lgr_knn.max_size - 1, 0,
                                      lgr_knn.max_size -1, (const float**)lgr_knn.in_data_distances);
        const uint32_t frame = *static_cast<const size_t*>(t[si_frame].get_dataptr());
        if (frame > (uint32_t)lgr_knn.fra_start && !lgr_knn.kNN_path.empty()) {
            char file_path[256];
            snprintf(file_path, sizeof(file_path), "%s/%05u.txt", lgr_knn.kNN_path.c_str(), frame);
            FILE* file = fopen(file_path, "a");
            fprintf(file, "#\n");

            kNN_data_t kNN_data = { (float**)lgr_knn.in_data_distances,
                                    (uint32_t**)lgr_knn.in_data_nearest,
#ifdef MOTION_ENABLE_DEBUG
                                    t[si_data_conflicts].get_dataptr<const uint32_t>(),
#else
                                    nullptr,
#endif
                                    lgr_knn.max_size };

            kNN_asso_conflicts_write(file, &kNN_data,
                                     t[si_RoIs0].get_dataptr<const RoI_t>(),
                                     *t[si_n_RoIs0].get_dataptr<const uint32_t>(),
                                     t[si_RoIs1].get_dataptr<const RoI_t>(),
                                     *t[si_n_RoIs1].get_dataptr<const uint32_t>());
            fclose(file);
        }
        return aff3ct::runtime::status_t::SUCCESS;
    });
}

Logger_kNN::~Logger_kNN() {
    free(this->in_data_nearest);
    free(this->in_data_distances);
}
