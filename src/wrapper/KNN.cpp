#include "motion/wrapper/KNN.hpp"

kNN::kNN(kNN_data_t* knn_data, int knn_k, uint32_t knn_d, float knn_s, uint32_t p_cca_roi_max2)
    : spu::module::Stateful(), knn_data(knn_data), knn_k(knn_k), knn_d(knn_d), knn_s(knn_s)
    
{
    const std::string name = "kNN";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("match");
    
    // kNN_match(knn_data, RoIs0, n_RoIs0, RoIs1, n_RoIs1, p_knn_k, p_knn_d, p_knn_s);

    //input socket
    size_t si_RoIs0 = this->template create_socket_in<uint8_t>(t, "in_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t si_RoIs1 = this->template create_socket_in<uint8_t>(t, "in_RoIs1", p_cca_roi_max2* sizeof(RoI_t));
    size_t si_n_RoIs0 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs0", 1);
    size_t si_n_RoIs1 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs1", 1);

    // output socket
    size_t so_RoIs0 = this->template create_socket_out<uint8_t>(t, "out_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t so_RoIs1 = this->template create_socket_out<uint8_t>(t, "out_RoIs1", p_cca_roi_max2* sizeof(RoI_t));
    
    // return value
    uint32_t so_n_assoc = this->template create_socket_out<uint32_t>(t, "out_n_assoc", 1);

    create_codelet(t, 
        [si_RoIs0, si_RoIs1, si_n_RoIs0, si_n_RoIs1, so_RoIs0, so_RoIs1, so_n_assoc]
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            kNN knn = static_cast<kNN&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t* n_RoIs0 = tsk[si_n_RoIs0].get_dataptr<const uint32_t>();
            const uint32_t* n_RoIs1 = tsk[si_n_RoIs1].get_dataptr<const uint32_t>();

            const RoI_t* RoIs0_in = tsk[si_RoIs0].get_dataptr<const RoI_t>();
            const RoI_t* RoIs1_in = tsk[si_RoIs1].get_dataptr<const RoI_t>();

            // Get the input and output data pointers from the task
            RoI_t* RoIs0_out = (RoI_t*)tsk[so_RoIs0].get_dataptr<uint8_t>();
            RoI_t* RoIs1_out = (RoI_t*)tsk[so_RoIs1].get_dataptr<uint8_t>();
            uint32_t* n_assoc_out = tsk[so_n_assoc].get_dataptr<uint32_t>();

            *n_assoc_out = kNN_match(knn.knn_data, (RoI_t*)RoIs0_in, *n_RoIs0, (RoI_t*)RoIs1_in, *n_RoIs1, knn.knn_k, knn.knn_d, knn.knn_s);

            // Copy the output data to the output socket
            memcpy(RoIs1_out, RoIs1_in, *n_RoIs1 * sizeof(RoI_t));
            memcpy(RoIs0_out, RoIs0_in, *n_RoIs0 * sizeof(RoI_t));
        
            return spu::runtime::status_t::SUCCESS;
        }
    );
}