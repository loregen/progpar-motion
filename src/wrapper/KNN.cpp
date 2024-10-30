#include "motion/wrapper/KNN.hpp"

kNN::kNN(kNN_data_t* knn_data, size_t n_RoIs0, size_t n_RoIs1,
            int knn_k, uint32_t knn_d, float knn_s)
    : spu::module::Stateful(),n_RoIs0(n_RoIs0), n_RoIs1(n_RoIs1), knn_data(knn_data), 
        knn_k(knn_k), knn_d(knn_d), knn_s(knn_s)
    
{
    const std::string name = "kNN";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("match");
    
    // kNN_match(knn_data, RoIs0, n_RoIs0, RoIs1, n_RoIs1, p_knn_k, p_knn_d, p_knn_s);

    // output socket
    size_t so_RoIs0 = this->template create_socket_out<uint8_t>(t, "out_RoIs0", n_RoIs0* sizeof(RoI_t));
    size_t so_RoIs1 = this->template create_socket_out<uint8_t>(t, "out_RoIs1", n_RoIs1* sizeof(RoI_t));
    
    // return value
    uint32_t so_n_assoc = this->template create_socket_out<uint32_t>(t, "out_n_assoc", 1);

    create_codelet(t, 
        [so_RoIs0, so_RoIs1, so_n_assoc] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            kNN knn = static_cast<kNN&>(m);
            
            // Get the input and output data pointers from the task
            RoI_t* RoIs0_out = (RoI_t*)tsk[so_RoIs0].get_dataptr<uint8_t>();
            RoI_t* RoIs1_out = (RoI_t*)tsk[so_RoIs1].get_dataptr<uint8_t>();
            uint32_t* n_assoc_out = tsk[so_n_assoc].get_dataptr<uint32_t>();

            *n_assoc_out = kNN_match(knn.knn_data, RoIs0_out, knn.n_RoIs0, RoIs1_out, knn.n_RoIs1, 
                                        knn.knn_k, knn.knn_d, knn.knn_s); 

            return spu::runtime::status_t::SUCCESS;
        }
    );
}