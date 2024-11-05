#ifndef KNN_HPP
#define KNN_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/kNN.h"   

class kNN : public spu::module::Stateful {
public:
    kNN(kNN_data_t* knn_data, int knn_k, uint32_t knn_d, float knn_s, uint32_t p_cca_roi_max2);
    void deep_copy(const kNN& m);
    kNN* clone() const;
private:
    kNN_data_t* knn_data;
    int knn_k;
    uint32_t knn_d;
    float knn_s;
};

#endif 