#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <nrc2.h>
#include <mipp.h>

#include "motion/macros.h"
#include "motion/morpho/morpho_compute.h"
void bitpack(const uint8_t **matrix_in,uint8_t ** matrix_out, const int i0, const int i1, const int j0, const int j1, const int  carry, const int ncolp)
{
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++)
    {
        int sub_index = 0;
        for(int j = j0; j<=j1 && sub_index<ncolp;j+=8)
        {
            uint8_t packed = 0;
            packed+= matrix_in[i][j + 0] ? 0b10000000: 0;
            packed+= matrix_in[i][j + 1] ? 0b01000000: 0;
            packed+= matrix_in[i][j + 2] ? 0b00100000: 0;
            packed+= matrix_in[i][j + 3] ? 0b00010000: 0;
            packed+= matrix_in[i][j + 4] ? 0b00001000: 0;
            packed+= matrix_in[i][j + 5] ? 0b00000100: 0;
            packed+= matrix_in[i][j + 6] ? 0b00000010: 0;
            packed+= matrix_in[i][j + 7] ? 0b00000001: 0;
            matrix_out[i][sub_index] = packed;
            sub_index++;
        }
        uint8_t packed = 0;
        int limit = carry == 0 ? 8:carry;
        for(int l = 0 ; l< limit; l++)
        {
            packed += matrix_in[i][j1-(j1%8) +l] ?  (0b10000000 >> l) :0;

        }
        matrix_out[i][ncolp] = packed;
    }
}

void bitunpack(const uint8_t **matrix_in, uint8_t **matrix_out, const int i0, const int i1, const int j0, const int j1, int  carry, const int ncolp)
{
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++)
    {
        int sub_index = 0;
        for(int j = j0; j<=j1 && sub_index<ncolp;j+=8)
        {
            matrix_out[i][j + 0] = (matrix_in[i][sub_index] & 0b10000000) ? 1: 0;
            matrix_out[i][j + 1] = (matrix_in[i][sub_index] & 0b01000000) ? 1: 0;
            matrix_out[i][j + 2] = (matrix_in[i][sub_index] & 0b00100000) ? 1: 0;
            matrix_out[i][j + 3] = (matrix_in[i][sub_index] & 0b00010000) ? 1: 0;
            matrix_out[i][j + 4] = (matrix_in[i][sub_index] & 0b00001000) ? 1: 0;
            matrix_out[i][j + 5] = (matrix_in[i][sub_index] & 0b00000100) ? 1: 0;
            matrix_out[i][j + 6] = (matrix_in[i][sub_index] & 0b00000010) ? 1: 0;
            matrix_out[i][j + 7] = (matrix_in[i][sub_index] & 0b00000001) ? 1: 0;
            sub_index++;
        }
        int limit = carry == 0 ? 8:carry;
        for(int l = 0 ; l< limit; l++)
        {
            matrix_out[i][j1-(j1%8) +l] = matrix_in[i][ncolp] & (0b10000000 >> l) ? 1:0;

        }
    }
}

morpho_data_t *morpho_alloc_data(const int i0, const int i1, const int j0, const int j1)
{
    morpho_data_t *morpho_data = (morpho_data_t *)malloc(sizeof(morpho_data_t));
    morpho_data->i0 = i0;
    morpho_data->i1 = i1;
    morpho_data->j0 = j0;
    morpho_data->j1 = j1;
    morpho_data->carry = (j1-j0)%8;
    morpho_data->ncolp = (j1-j0+7)/8;
    morpho_data->IB = ui8matrix(morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    morpho_data->IB_packed = ui8matrix(morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp);
    morpho_data->IN_packed = ui8matrix(morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp);
    return morpho_data;
}

void morpho_init_data(morpho_data_t *morpho_data)
{
    zero_ui8matrix(morpho_data->IB, morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    zero_ui8matrix(morpho_data->IB_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp);
    zero_ui8matrix(morpho_data->IN_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp);
}

void morpho_free_data(morpho_data_t *morpho_data)
{
    free_ui8matrix(morpho_data->IB, morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    free_ui8matrix(morpho_data->IB_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp);
    free_ui8matrix(morpho_data->IN_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp);
    free(morpho_data);
}

void morpho_compute_erosion3(const uint8_t **img_in, uint8_t **img_out, const int i0, const int i1, const int j0,
                             const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    assert(img_in != (const uint8_t **)img_out);

    /* copy the borders:
        1st loop: address non contiguos in memory_makes no sense to vectorize here
    */

    const uint8_t vector_size = mipp::N<uint8_t>();
    auto vec_loop_size = (j1 / mipp::N<uint8_t>()) * mipp::N<uint8_t>();
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++)
    {
        img_out[i][j0] = img_in[i][j0];
        img_out[i][j1] = img_in[i][j1];
    }

    #pragma omp parallel for
    for (int j = j0; j < vec_loop_size; j += vector_size)
    {
        mipp::Reg<uint8_t>(&img_in[i0][j]).store(&img_out[i0][j]);
        mipp::Reg<uint8_t>(&img_in[i1][j]).store(&img_out[i1][j]);
    }
    for (int j = vec_loop_size; j <= j1; j++)
    {
        img_out[i0][j] = img_in[i0][j];
        img_out[i1][j] = img_in[i1][j];
    }

    vec_loop_size = ((j1 - 1) / mipp::N<uint8_t>()) * mipp::N<uint8_t>();
    #pragma omp parallel for
    for (int i = i0 + 1; i <= i1 - 1; i++)
    {
        for (int j = j0 + 1; j < vec_loop_size; j += vector_size)
        {
            mipp::Reg<uint8_t> c0 = mipp::Reg<uint8_t>(&img_in[i - 1][j - 1]) & mipp::Reg<uint8_t>(&img_in[i - 1][j]) & mipp::Reg<uint8_t>(&img_in[i - 1][j + 1]);
            mipp::Reg<uint8_t> c1 = mipp::Reg<uint8_t>(&img_in[i + 0][j - 1]) & mipp::Reg<uint8_t>(&img_in[i + 0][j]) & mipp::Reg<uint8_t>(&img_in[i + 0][j + 1]);
            mipp::Reg<uint8_t> c2 = mipp::Reg<uint8_t>(&img_in[i + 1][j - 1]) & mipp::Reg<uint8_t>(&img_in[i + 1][j]) & mipp::Reg<uint8_t>(&img_in[i + 1][j + 1]);
            (c0 & c1 & c2).store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j <= j1 - 1; j++)
        {
            uint8_t c0 = img_in[i - 1][j - 1] & img_in[i - 1][j] & img_in[i - 1][j + 1];
            uint8_t c1 = img_in[i + 0][j - 1] & img_in[i + 0][j] & img_in[i + 0][j + 1];
            uint8_t c2 = img_in[i + 1][j - 1] & img_in[i + 1][j] & img_in[i + 1][j + 1];
            img_out[i][j] = c0 & c1 & c2;
        }
    }
}

void morpho_compute_dilation3(const uint8_t **img_in, uint8_t **img_out, const int i0, const int i1, const int j0,
                              const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    assert(img_in != (const uint8_t **)img_out);

    /* copy the borders:
        1st loop: address non contiguos in memory_makes no sense to vectorize here
    */
    const uint8_t vector_size = mipp::N<uint8_t>();
    auto vec_loop_size = (j1 / mipp::N<uint8_t>()) * mipp::N<uint8_t>();

#pragma omp parallel for
    for (int i = i0; i <= i1; i++)
    {
        img_out[i][j0] = img_in[i][j0];
        img_out[i][j1] = img_in[i][j1];
    }

#pragma omp parallel for
    for (int j = j0; j < vec_loop_size; j += vector_size)
    {
        mipp::Reg<uint8_t>(&img_in[i0][j]).store(&img_out[i0][j]);
        mipp::Reg<uint8_t>(&img_in[i1][j]).store(&img_out[i1][j]);
    }
    for (int j = vec_loop_size; j <= j1; j++)
    {
        img_out[i0][j] = img_in[i0][j];
        img_out[i1][j] = img_in[i1][j];
    }

    vec_loop_size = ((j1 - 1) / mipp::N<uint8_t>()) * mipp::N<uint8_t>();
#pragma omp parallel for
    for (int i = i0 + 1; i <= i1 - 1; i++)
    {
        for (int j = j0 + 1; j < vec_loop_size; j += vector_size)
        {
            mipp::Reg<uint8_t> c0 = mipp::Reg<uint8_t>(&img_in[i - 1][j - 1]) | mipp::Reg<uint8_t>(&img_in[i - 1][j]) | mipp::Reg<uint8_t>(&img_in[i - 1][j + 1]);
            mipp::Reg<uint8_t> c1 = mipp::Reg<uint8_t>(&img_in[i + 0][j - 1]) | mipp::Reg<uint8_t>(&img_in[i + 0][j]) | mipp::Reg<uint8_t>(&img_in[i + 0][j + 1]);
            mipp::Reg<uint8_t> c2 = mipp::Reg<uint8_t>(&img_in[i + 1][j - 1]) | mipp::Reg<uint8_t>(&img_in[i + 1][j]) | mipp::Reg<uint8_t>(&img_in[i + 1][j + 1]);
            (c0 | c1 | c2).store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j <= j1 - 1; j++)
        {
            uint8_t c0 = img_in[i - 1][j - 1] | img_in[i - 1][j] | img_in[i - 1][j + 1];
            uint8_t c1 = img_in[i + 0][j - 1] | img_in[i + 0][j] | img_in[i + 0][j + 1];
            uint8_t c2 = img_in[i + 1][j - 1] | img_in[i + 1][j] | img_in[i + 1][j + 1];
            img_out[i][j] = c0 | c1 | c2;
        }
    }
}

void morpho_compute_opening3(morpho_data_t *morpho_data, const uint8_t **img_in, uint8_t **img_out, const int i0,
                             const int i1, const int j0, const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    
    morpho_compute_erosion3((const uint8_t **)img_in, morpho_data->IB, i0, i1, j0, j1);
    morpho_compute_dilation3((const uint8_t **)morpho_data->IB, img_out, i0, i1, j0, j1);
    bitpack(img_in,morpho_data->IB_packed,i0,i1,j0,j1, morpho_data->carry,morpho_data->ncolp);
    bitunpack((const uint8_t **)morpho_data->IB_packed,img_out,i0,i1,j0,j1,morpho_data->carry,morpho_data->ncolp);
}

void morpho_compute_closing3(morpho_data_t *morpho_data, const uint8_t **img_in, uint8_t **img_out, const int i0,
                             const int i1, const int j0, const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    morpho_compute_dilation3((const uint8_t **)img_in, morpho_data->IB, i0, i1, j0, j1);
    morpho_compute_erosion3((const uint8_t **)morpho_data->IB, img_out, i0, i1, j0, j1);
}
