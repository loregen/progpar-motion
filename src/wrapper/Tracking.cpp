#include "motion/wrapper/Tracking.hpp"

Tracking::Tracking(tracking_data_t* tracking_data, size_t n_RoIs, size_t frame, size_t r_extrapol, 
            size_t fra_obj_min, uint8_t save_RoIs_id, uint8_t extrapol_order_max, 
            float min_extrapol_ratio_S)
    : spu::module::Stateful(), tracking_data(tracking_data), n_RoIs(n_RoIs), frame(frame), r_extrapol(r_extrapol), 
        fra_obj_min(fra_obj_min), save_RoIs_id(save_RoIs_id), extrapol_order_max(extrapol_order_max), 
        min_extrapol_ratio_S(min_extrapol_ratio_S)
    
{
    const std::string name = "Tracking";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("perform");
  
    // output socket
    size_t so_RoIs = this->template create_socket_in<uint8_t>(t, "in_RoIs", n_RoIs* sizeof(RoI_t));
        
    create_codelet(t, 
        [so_RoIs] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Tracking tracking = static_cast<Tracking&>(m);
            
            // Get the input and output data pointers from the task
            const RoI_t* RoIs_in = (RoI_t*)tsk[so_RoIs].get_dataptr<const uint8_t>();

            tracking_perform(tracking.tracking_data, RoIs_in, 
                        tracking.n_RoIs, tracking.frame, tracking.r_extrapol, 
                        tracking.fra_obj_min, tracking.save_RoIs_id, tracking.extrapol_order_max,
                        tracking.min_extrapol_ratio_S);

            return spu::runtime::status_t::SUCCESS;
        }
    );
}