#include <math.h>
#include <stdlib.h>
#include <nrc2.h>
#include <omp.h>
#include <mipp.h>

#include "motion/macros.h"
#include "motion/sigma_delta/sigma_delta_compute.h"

sigma_delta_data_t* sigma_delta_alloc_data(const int i0, const int i1, const int j0, const int j1, const uint8_t vmin,
                                           const uint8_t vmax) {
    sigma_delta_data_t* sd_data = (sigma_delta_data_t*)malloc(sizeof(sigma_delta_data_t));
    sd_data->i0 = i0;
    sd_data->i1 = i1;
    sd_data->j0 = j0;
    sd_data->j1 = j1;
    sd_data->vmin = vmin;
    sd_data->vmax = vmax;
    sd_data->M = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    sd_data->O = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    sd_data->V = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    return sd_data;
}

void sigma_delta_init_data(sigma_delta_data_t* sd_data, const uint8_t** img_in, const int i0, const int i1,
                           const int j0, const int j1) {
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            sd_data->M[i][j] = img_in != NULL ? img_in[i][j] : sd_data->vmax;
            sd_data->V[i][j] = sd_data->vmin;
        }
    }
}

void sigma_delta_free_data(sigma_delta_data_t* sd_data) {
    free_ui8matrix(sd_data->M, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free_ui8matrix(sd_data->O, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free_ui8matrix(sd_data->V, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free(sd_data);
}

void sigma_delta_compute(sigma_delta_data_t *sd_data, const uint8_t** img_in, uint8_t** img_out, const int i0,
                         const int i1, const int j0, const int j1, const uint8_t N) {
    
    const uint8_t vector_size = mipp::N<uint8_t>();
    auto vec_loop_size = (j1 / mipp::N<uint8_t>()) * mipp::N<uint8_t>();
            
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <vec_loop_size; j+=vector_size) {
            mipp::Reg<uint8_t> new_m = &sd_data->M[i][j];
            mipp::Msk<vector_size> msk0 = mipp::Reg<uint8_t>(&(sd_data->M[i][j]))< mipp::Reg<uint8_t>(&img_in[i][j]);
            mipp::Msk<vector_size> msk1 = mipp::Reg<uint8_t>(&(sd_data->M[i][j]))> mipp::Reg<uint8_t>(&img_in[i][j]);
            new_m = new_m+(msk0.toReg<uint8_t>()&mipp::Reg<uint8_t>(1));
            new_m = new_m-(msk1.toReg<uint8_t>()&mipp::Reg<uint8_t>(1));
            new_m.store(&sd_data->M[i][j]);
        }
        for (int j = vec_loop_size; j <= j1; j++) {
            uint8_t new_m = sd_data->M[i][j];
            if (sd_data->M[i][j] < img_in[i][j])
                new_m += 1;
            else if (sd_data->M[i][j] > img_in[i][j])
                new_m -= 1;
            sd_data->M[i][j] = new_m;
        }
    }

    #pragma omp parallel for
    for (int i = i0; i <= i1; i++) {
        for (int j = 0; j < vec_loop_size; j+=vector_size) {
            mipp::Msk<vector_size> msk = mipp::Reg<uint8_t>(&(sd_data->M[i][j])) > mipp::Reg<uint8_t>(&img_in[i][j]);
            mipp::Reg<uint8_t> partial_0 = mipp::blend<uint8_t>(mipp::Reg<uint8_t>(&(sd_data->M[i][j])),mipp::Reg<uint8_t>(&img_in[i][j]),msk);
            mipp::Reg<uint8_t> partial_1 = mipp::blend<uint8_t>(mipp::Reg<uint8_t>(&img_in[i][j]),mipp::Reg<uint8_t>(&(sd_data->M[i][j])),msk);
            (partial_0-partial_1).store(&(sd_data->O[i][j]));
        }
        for (int j = vec_loop_size; j <= j1; j++) {
            sd_data->O[i][j] = abs(sd_data->M[i][j] - img_in[i][j]);
        }
    }

    if(N!=2)
    {
        fprintf(stderr, " N parameter !=2 deactivating manual vectorization ");
        #pragma omp parallel for
        for (int i = i0; i <= i1; i++) {
            for (int j = i0; j <= j1; j++) {
                uint8_t new_v = sd_data->V[i][j];
                if (sd_data->V[i][j] < N * sd_data->O[i][j])
                    new_v += 1;
                else if (sd_data->V[i][j] > N * sd_data->O[i][j])
                    new_v -= 1;
                sd_data->V[i][j] = MAX(MIN(new_v, sd_data->vmax), sd_data->vmin);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int i = i0; i <= i1; i++) {
            for (int j = i0; j <vec_loop_size; j+=vector_size) {

                mipp::Reg<uint8_t> new_v = &sd_data->V[i][j];
                mipp::Msk<vector_size> msk0 = mipp::Reg<uint8_t>(&(sd_data->V[i][j]))< mipp::Reg<uint8_t>(&(sd_data->O[i][j]))+mipp::Reg<uint8_t>(&(sd_data->O[i][j]));
                mipp::Msk<vector_size> msk1 = mipp::Reg<uint8_t>(&(sd_data->V[i][j]))> mipp::Reg<uint8_t>(&(sd_data->O[i][j]))+mipp::Reg<uint8_t>(&(sd_data->O[i][j]));
                new_v = new_v+(msk0.toReg<uint8_t>()&mipp::Reg<uint8_t>(1));
                new_v = new_v-(msk1.toReg<uint8_t>()&mipp::Reg<uint8_t>(1));
                mipp::max(mipp::min(new_v, mipp::Reg<uint8_t>(sd_data->vmax)), mipp::Reg<uint8_t>(sd_data->vmin)).store(&(sd_data->V[i][j]));
            }
            for (int j = vec_loop_size; j <= j1; j++) {
                uint8_t new_v = sd_data->V[i][j];
                if (sd_data->V[i][j] < N * sd_data->O[i][j])
                    new_v += 1;
                else if (sd_data->V[i][j] > N * sd_data->O[i][j])
                    new_v -= 1;
                sd_data->V[i][j] = MAX(MIN(new_v, sd_data->vmax), sd_data->vmin);
            }
        }
        
    }
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++) {
        for (int j = 0; j < j1; j+=vector_size) {
           (mipp::Reg<uint8_t>(&(sd_data->O[i][j])) >= mipp::Reg<uint8_t>(&(sd_data->V[i][j]))).toReg<uint8_t>().store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j <= j1; j++) {
            img_out[i][j] = sd_data->O[i][j] < sd_data->V[i][j] ? 0 : 255;
        }
    }
}
