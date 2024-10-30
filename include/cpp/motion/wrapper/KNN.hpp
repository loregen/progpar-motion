#ifndef KNN_HPP
#define KNN_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/kNN.h"   

class kNN : public spu::module::Stateful {
public:
    kNN(kNN_data_t* knn_data, size_t n_RoIs0, size_t n_RoIs1,
            int knn_k, uint32_t knn_d, float knn_s);

private:
    kNN_data_t* knn_data;
    size_t n_RoIs0, n_RoIs1;
    int knn_k;
    uint32_t knn_d;
    float knn_s;
};

#endif 