#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <nrc2.h>
#include <mipp.h>

#include "motion/macros.h"
#include "motion/morpho/morpho_compute.h"

void print_uint8_t(uint8_t n)
{
    int i;
    for (i = 8; i >= 0; i--)
        if (i != 8)
            printf("%d", (n & (1 << i)) >> i);
    putchar('\n');
}

void set_bit_from_source(uint8_t *target, char source, int n)
{
    // Extract the nth bit from source
    uint8_t nth_bit = (source >> n) & 1;

    // Clear the nth bit of target
    *target &= ~(1 << n);

    // Set the nth bit of target to the nth bit of source
    *target |= (nth_bit << n);
}

void bitpack(const uint8_t **matrix_in, uint8_t **matrix_out, const int i0, const int i1, const int j0, const int j1, const int carry, const int ncolp)
{
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++)
    {
        int subidx = 0;

        for (int j = j0; j <= j1; j += 8)
        {
            uint8_t packed = 0;
            packed += (matrix_in[i][j + 0] ? 0b10000000 : 0);
            packed += (matrix_in[i][j + 1] ? 0b01000000 : 0);
            packed += (matrix_in[i][j + 2] ? 0b00100000 : 0);
            packed += (matrix_in[i][j + 3] ? 0b00010000 : 0);
            packed += (matrix_in[i][j + 4] ? 0b00001000 : 0);
            packed += (matrix_in[i][j + 5] ? 0b00000100 : 0);
            packed += (matrix_in[i][j + 6] ? 0b00000010 : 0);
            packed += (matrix_in[i][j + 7] ? 0b00000001 : 0);
            matrix_out[i][subidx] = packed;
            subidx += 1;
        }
    }
}

void bitunpack(const uint8_t **matrix_in, uint8_t **matrix_out, const int i0, const int i1, const int j0, const int j1, int carry, const int ncolp)
{
    #pragma omp parallel for
    for (int i = i0; i <= i1; i++)
    {
        int subidx = 0;
        for (int j = j0; j <= j1; j += 8)
        {
            matrix_out[i][j + 0] = ((matrix_in[i][subidx] & 0b10000000) > 0 ? 255 : 0);
            matrix_out[i][j + 1] = ((matrix_in[i][subidx] & 0b01000000) > 0 ? 255 : 0);
            matrix_out[i][j + 2] = ((matrix_in[i][subidx] & 0b00100000) > 0 ? 255 : 0);
            matrix_out[i][j + 3] = ((matrix_in[i][subidx] & 0b00010000) > 0 ? 255 : 0);
            matrix_out[i][j + 4] = ((matrix_in[i][subidx] & 0b00001000) > 0 ? 255 : 0);
            matrix_out[i][j + 5] = ((matrix_in[i][subidx] & 0b00000100) > 0 ? 255 : 0);
            matrix_out[i][j + 6] = ((matrix_in[i][subidx] & 0b00000010) > 0 ? 255 : 0);
            matrix_out[i][j + 7] = ((matrix_in[i][subidx] & 0b00000001) > 0 ? 255 : 0);
            subidx += 1;
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
    morpho_data->carry = (j1 - j0 + 1) % 8;
    morpho_data->ncolp = (j1 - j0 + 1) / 8 + (morpho_data->carry == 0 ? 0 : 1);
    morpho_data->IB = ui8matrix(morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    morpho_data->IB_packed = ui8matrix(morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp - 1);
    morpho_data->IN_packed = ui8matrix(morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp - 1);
    return morpho_data;
}

void morpho_init_data(morpho_data_t *morpho_data)
{
    zero_ui8matrix(morpho_data->IB, morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    zero_ui8matrix(morpho_data->IB_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp - 1);
    zero_ui8matrix(morpho_data->IN_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp - 1);
}

void morpho_free_data(morpho_data_t *morpho_data)
{
    free_ui8matrix(morpho_data->IB, morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    free_ui8matrix(morpho_data->IB_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp - 1);
    free_ui8matrix(morpho_data->IN_packed, morpho_data->i0, morpho_data->i1, 0, morpho_data->ncolp - 1);
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

            mipp::Reg<uint8_t> c0 = mipp::Reg<uint8_t>(&img_in[i - 1][j - 1]) & mipp::Reg<uint8_t>(&img_in[i][j - 1]) & mipp::Reg<uint8_t>(&img_in[i + 1][j - 1]);
            mipp::Reg<uint8_t> c1 = mipp::Reg<uint8_t>(&img_in[i - 1][j + 0]) & mipp::Reg<uint8_t>(&img_in[i][j + 0]) & mipp::Reg<uint8_t>(&img_in[i + 1][j + 0]);
            mipp::Reg<uint8_t> c2 = mipp::Reg<uint8_t>(&img_in[i - 1][j + 1]) & mipp::Reg<uint8_t>(&img_in[i][j + 1]) & mipp::Reg<uint8_t>(&img_in[i + 1][j + 1]);
            (c0 & c1 & c2).store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j <= j1 - 1; j++)
        {
            uint8_t c0 = img_in[i - 1][j - 1] & img_in[i][j - 1] & img_in[i + 1][j - 1];
            uint8_t c1 = img_in[i - 1][j + 0] & img_in[i][j + 0] & img_in[i + 1][j + 0];
            uint8_t c2 = img_in[i - 1][j + 1] & img_in[i][j + 1] & img_in[i + 1][j + 1];

            img_out[i][j] = c0 & c1 & c2;
        }
    }
}
void morpho_compute_erosion3_packed(const uint8_t **img_in, uint8_t **img_out, const int i0, const int i1, const int j0,
                                    const int j1, const int carry, const int ncols)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    assert(img_in != (const uint8_t **)img_out);

    /* copy the borders:
        1st loop: address non contiguos in memory_makes no sense to vectorize here
    */

    const uint8_t vector_size = mipp::N<uint8_t>();
    auto vec_loop_size = (ncols / mipp::N<uint8_t>()) * mipp::N<uint8_t>();

#pragma omp parallel for
    for (int j = 0; j < vec_loop_size; j += vector_size)
    {
        mipp::Reg<uint8_t>(&img_in[i0][j]).store(&img_out[i0][j]);
        mipp::Reg<uint8_t>(&img_in[i1][j]).store(&img_out[i1][j]);
    }
    for (int j = vec_loop_size; j < ncols; j++)
    {
        img_out[i0][j] = img_in[i0][j];
        img_out[i1][j] = img_in[i1][j];
    }

    vec_loop_size = ((ncols - 1) / mipp::N<uint8_t>()) * mipp::N<uint8_t>();
#pragma omp parallel for
    for (int i = i0 + 1; i <= i1 - 1; i++)
    {

        uint8_t rest_i_m_j_p = ((img_in[i - 1][1] & 0b10000000) >> 7);
        uint8_t rest_i_n_j_p = ((img_in[i - 0][1] & 0b10000000) >> 7);
        uint8_t rest_i_p_j_p = ((img_in[i + 1][1] & 0b10000000) >> 7);

        uint8_t c0 = ((img_in[i - 1][0] >> 1)) & ((img_in[i + 0][0] >> 1)) & ((img_in[i + 1][0] >> 1));
        uint8_t c1 = (img_in[i - 1][0]) & (img_in[i + 0][0]) & (img_in[i + 1][0]);
        uint8_t c2 = ((img_in[i - 1][0] << 1) + rest_i_m_j_p) & ((img_in[i + 0][0] << 1) + rest_i_n_j_p) & ((img_in[i + 1][0] << 1) + rest_i_p_j_p);

        img_out[i][0] = (c0 & c1 & c2) + (img_in[i][0] & 0b10000000);

        uint8_t rest_i_m_j_m = ((img_in[i - 1][ncols - 2] & 0b00000001) << 7);
        uint8_t rest_i_n_j_m = ((img_in[i - 0][ncols - 2] & 0b00000001) << 7);
        uint8_t rest_i_p_j_m = ((img_in[i + 1][ncols - 2] & 0b00000001) << 7);

        c0 = ((img_in[i - 1][ncols - 1] >> 1) + rest_i_m_j_m) & ((img_in[i + 0][ncols - 1] >> 1) + rest_i_n_j_m) & ((img_in[i + 1][ncols - 1] >> 1) + rest_i_p_j_m);
        c1 = (img_in[i - 1][ncols - 1]) & (img_in[i + 0][ncols - 1]) & (img_in[i + 1][ncols - 1]);
        c2 = ((img_in[i - 1][ncols - 1] << 1)) & ((img_in[i + 0][ncols - 1] << 1)) & (img_in[i + 1][ncols - 1] << 1);

        img_out[i][ncols - 1] = (c0 & c1 & c2) + (img_in[i][ncols - 1] & 0b00000001);

        for (int j = 1; j < vec_loop_size; j += vector_size)
        {
            mipp::Reg<uint8_t> rest_i_m_j_m = ((mipp::Reg<uint8_t>(&img_in[i - 1][j - 1]) & 0b00000001) << 7);
            mipp::Reg<uint8_t> rest_i_n_j_m = ((mipp::Reg<uint8_t>(&img_in[i - 0][j - 1]) & 0b00000001) << 7);
            mipp::Reg<uint8_t> rest_i_p_j_m = ((mipp::Reg<uint8_t>(&img_in[i + 1][j - 1]) & 0b00000001) << 7);

            mipp::Reg<uint8_t> rest_i_m_j_p = ((mipp::Reg<uint8_t>(&img_in[i - 1][j + 1]) & 0b10000000) >> 7);
            mipp::Reg<uint8_t> rest_i_n_j_p = ((mipp::Reg<uint8_t>(&img_in[i - 0][j + 1]) & 0b10000000) >> 7);
            mipp::Reg<uint8_t> rest_i_p_j_p = ((mipp::Reg<uint8_t>(&img_in[i + 1][j + 1]) & 0b10000000) >> 7);

            mipp::Reg<uint8_t> c0 = ((mipp::Reg<uint8_t>(&img_in[i - 1][j]) >> 1) + rest_i_m_j_m) & ((mipp::Reg<uint8_t>(&img_in[i + 0][j]) >> 1) + rest_i_n_j_m) & ((mipp::Reg<uint8_t>(&img_in[i + 1][j]) >> 1) + rest_i_p_j_m);
            mipp::Reg<uint8_t> c1 = (mipp::Reg<uint8_t>(&img_in[i - 1][j])) & (mipp::Reg<uint8_t>(&img_in[i + 0][j])) & (mipp::Reg<uint8_t>(&img_in[i + 1][j]));
            mipp::Reg<uint8_t> c2 = ((mipp::Reg<uint8_t>(&img_in[i - 1][j]) << 1) + rest_i_m_j_p) & ((mipp::Reg<uint8_t>(&img_in[i + 0][j]) << 1) + rest_i_n_j_p) & ((mipp::Reg<uint8_t>(&img_in[i + 1][j]) << 1) + rest_i_p_j_p);
            (c0 & c1 & c2).store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j < ncols - 1; j++)
        {
            uint8_t rest_i_m_j_m = ((img_in[i - 1][j - 1] & 0b00000001) << 7);
            uint8_t rest_i_n_j_m = ((img_in[i - 0][j - 1] & 0b00000001) << 7);
            uint8_t rest_i_p_j_m = ((img_in[i + 1][j - 1] & 0b00000001) << 7);

            uint8_t rest_i_m_j_p = ((img_in[i - 1][j + 1] & 0b10000000) >> 7);
            uint8_t rest_i_n_j_p = ((img_in[i - 0][j + 1] & 0b10000000) >> 7);
            uint8_t rest_i_p_j_p = ((img_in[i + 1][j + 1] & 0b10000000) >> 7);

            uint8_t c0 = ((img_in[i - 1][j] >> 1) + rest_i_m_j_m) & ((img_in[i + 0][j] >> 1) + rest_i_n_j_m) & ((img_in[i + 1][j] >> 1) + rest_i_p_j_m);
            uint8_t c1 = (img_in[i - 1][j]) & (img_in[i + 0][j]) & (img_in[i + 1][j]);
            uint8_t c2 = ((img_in[i - 1][j] << 1) + rest_i_m_j_p) & ((img_in[i + 0][j] << 1) + rest_i_n_j_p) & ((img_in[i + 1][j] << 1) + rest_i_p_j_p);
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
            mipp::Reg<uint8_t> c0 = mipp::Reg<uint8_t>(&img_in[i - 1][j - 1]) | mipp::Reg<uint8_t>(&img_in[i][j - 1]) | mipp::Reg<uint8_t>(&img_in[i + 1][j - 1]);
            mipp::Reg<uint8_t> c1 = mipp::Reg<uint8_t>(&img_in[i - 1][j + 0]) | mipp::Reg<uint8_t>(&img_in[i][j + 0]) | mipp::Reg<uint8_t>(&img_in[i + 1][j + 0]);
            mipp::Reg<uint8_t> c2 = mipp::Reg<uint8_t>(&img_in[i - 1][j + 1]) | mipp::Reg<uint8_t>(&img_in[i][j + 1]) | mipp::Reg<uint8_t>(&img_in[i + 1][j + 1]);
            (c0 | c1 | c2).store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j <= j1 - 1; j++)
        {
            uint8_t c0 = img_in[i - 1][j - 1] | img_in[i][j - 1] | img_in[i + 1][j - 1];
            uint8_t c1 = img_in[i - 1][j + 0] | img_in[i][j + 0] | img_in[i + 1][j + 0];
            uint8_t c2 = img_in[i - 1][j + 1] | img_in[i][j + 1] | img_in[i + 1][j + 1];
            img_out[i][j] = c0 | c1 | c2;
        }
    }
}
void morpho_compute_dilation3_packed(const uint8_t **img_in, uint8_t **img_out, const int i0, const int i1, const int j0,
                                    const int j1, const int carry, const int ncols)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    assert(img_in != (const uint8_t **)img_out);

    /* copy the borders:
        1st loop: address non contiguos in memory_makes no sense to vectorize here
    */

    const uint8_t vector_size = mipp::N<uint8_t>();
    auto vec_loop_size = (ncols / mipp::N<uint8_t>()) * mipp::N<uint8_t>();

#pragma omp parallel for
    for (int j = 0; j < vec_loop_size; j += vector_size)
    {
        mipp::Reg<uint8_t>(&img_in[i0][j]).store(&img_out[i0][j]);
        mipp::Reg<uint8_t>(&img_in[i1][j]).store(&img_out[i1][j]);
    }
    for (int j = vec_loop_size; j < ncols; j++)
    {
        img_out[i0][j] = img_in[i0][j];
        img_out[i1][j] = img_in[i1][j];
    }

    vec_loop_size = ((ncols - 1) / mipp::N<uint8_t>()) * mipp::N<uint8_t>();
#pragma omp parallel for
    for (int i = i0 + 1; i <= i1 - 1; i++)
    {

        uint8_t rest_i_m_j_p = ((img_in[i - 1][1] & 0b10000000) >> 7);
        uint8_t rest_i_n_j_p = ((img_in[i - 0][1] & 0b10000000) >> 7);
        uint8_t rest_i_p_j_p = ((img_in[i + 1][1] & 0b10000000) >> 7);

        uint8_t c0 = ((img_in[i - 1][0] >> 1))                      | ((img_in[i + 0][0] >> 1))                | ((img_in[i + 1][0] >> 1));
        uint8_t c1 = (img_in[i - 1][0])                             | (img_in[i + 0][0])                       | (img_in[i + 1][0]);
        uint8_t c2 = ((img_in[i - 1][0] << 1) + rest_i_m_j_p)       | ((img_in[i + 0][0] << 1) + rest_i_n_j_p) | ((img_in[i + 1][0] << 1) + rest_i_p_j_p);

        img_out[i][0] = ((c0 | c1 | c2) & 0b01111111) + (img_in[i][0] & 0b10000000);

        uint8_t rest_i_m_j_m = ((img_in[i - 1][ncols - 2] & 0b00000001) << 7);
        uint8_t rest_i_n_j_m = ((img_in[i - 0][ncols - 2] & 0b00000001) << 7);
        uint8_t rest_i_p_j_m = ((img_in[i + 1][ncols - 2] & 0b00000001) << 7);

        c0 = ((img_in[i - 1][ncols - 1] >> 1) + rest_i_m_j_m) | ((img_in[i + 0][ncols - 1] >> 1) + rest_i_n_j_m) | ((img_in[i + 1][ncols - 1] >> 1) + rest_i_p_j_m);
        c1 = (img_in[i - 1][ncols - 1])                       | (img_in[i + 0][ncols - 1])                       | (img_in[i + 1][ncols - 1]);
        c2 = ((img_in[i - 1][ncols - 1] << 1))                | ((img_in[i + 0][ncols - 1] << 1))                | (img_in[i + 1][ncols - 1] << 1);

        img_out[i][ncols - 1] = ((c0 | c1 | c2)& 0b11111110) + (img_in[i][ncols - 1] & 0b00000001);

        for (int j = 1; j < vec_loop_size; j += vector_size)
        {
            mipp::Reg<uint8_t> rest_i_m_j_m = ((mipp::Reg<uint8_t>(&img_in[i - 1][j - 1]) & 0b00000001) << 7);
            mipp::Reg<uint8_t> rest_i_n_j_m = ((mipp::Reg<uint8_t>(&img_in[i - 0][j - 1]) & 0b00000001) << 7);
            mipp::Reg<uint8_t> rest_i_p_j_m = ((mipp::Reg<uint8_t>(&img_in[i + 1][j - 1]) & 0b00000001) << 7);

            mipp::Reg<uint8_t> rest_i_m_j_p = ((mipp::Reg<uint8_t>(&img_in[i - 1][j + 1]) & 0b10000000) >> 7);
            mipp::Reg<uint8_t> rest_i_n_j_p = ((mipp::Reg<uint8_t>(&img_in[i - 0][j + 1]) & 0b10000000) >> 7);
            mipp::Reg<uint8_t> rest_i_p_j_p = ((mipp::Reg<uint8_t>(&img_in[i + 1][j + 1]) & 0b10000000) >> 7);

            mipp::Reg<uint8_t> c0 = ((mipp::Reg<uint8_t>(&img_in[i - 1][j]) >> 1) + rest_i_m_j_m) | ((mipp::Reg<uint8_t>(&img_in[i + 0][j]) >> 1) + rest_i_n_j_m) | ((mipp::Reg<uint8_t>(&img_in[i + 1][j]) >> 1) + rest_i_p_j_m);
            mipp::Reg<uint8_t> c1 = (mipp::Reg<uint8_t>(&img_in[i - 1][j]))                       | (mipp::Reg<uint8_t>(&img_in[i + 0][j]))                       | (mipp::Reg<uint8_t>(&img_in[i + 1][j]));
            mipp::Reg<uint8_t> c2 = ((mipp::Reg<uint8_t>(&img_in[i - 1][j]) << 1) + rest_i_m_j_p) | ((mipp::Reg<uint8_t>(&img_in[i + 0][j]) << 1) + rest_i_n_j_p) | ((mipp::Reg<uint8_t>(&img_in[i + 1][j]) << 1) + rest_i_p_j_p);
            (c0 | c1 | c2).store(&img_out[i][j]);
        }
        for (int j = vec_loop_size; j < ncols - 1; j++)
        {
            uint8_t rest_i_m_j_m = ((img_in[i - 1][j - 1] & 0b00000001) << 7);
            uint8_t rest_i_n_j_m = ((img_in[i - 0][j - 1] & 0b00000001) << 7);
            uint8_t rest_i_p_j_m = ((img_in[i + 1][j - 1] & 0b00000001) << 7);

            uint8_t rest_i_m_j_p = ((img_in[i - 1][j + 1] & 0b10000000) >> 7);
            uint8_t rest_i_n_j_p = ((img_in[i - 0][j + 1] & 0b10000000) >> 7);
            uint8_t rest_i_p_j_p = ((img_in[i + 1][j + 1] & 0b10000000) >> 7);

            uint8_t c0 = ((img_in[i - 1][j] >> 1) + rest_i_m_j_m) | ((img_in[i + 0][j] >> 1) + rest_i_n_j_m) | ((img_in[i + 1][j] >> 1) + rest_i_p_j_m);
            uint8_t c1 = (img_in[i - 1][j])                       | (img_in[i + 0][j])                       | (img_in[i + 1][j]);
            uint8_t c2 = ((img_in[i - 1][j] << 1) + rest_i_m_j_p) | ((img_in[i + 0][j] << 1) + rest_i_n_j_p) | ((img_in[i + 1][j] << 1) + rest_i_p_j_p);
            img_out[i][j] = c0 | c1 | c2;
        }
    }
}
void morpho_compute_opening3(morpho_data_t *morpho_data, const uint8_t **img_in, uint8_t **img_out, const int i0,
                             const int i1, const int j0, const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);

    bitpack((const uint8_t **)img_in, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_erosion3_packed((const uint8_t **)morpho_data->IN_packed, morpho_data->IB_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_dilation3_packed((const uint8_t **)morpho_data->IB_packed, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    bitunpack((const uint8_t **)morpho_data->IN_packed, img_out, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
}

void morpho_compute_closing3(morpho_data_t *morpho_data, const uint8_t **img_in, uint8_t **img_out, const int i0,
                             const int i1, const int j0, const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    bitpack((const uint8_t **)img_in, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_dilation3_packed((const uint8_t **)morpho_data->IN_packed, morpho_data->IB_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_erosion3_packed((const uint8_t **)morpho_data->IB_packed, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    bitunpack((const uint8_t **)morpho_data->IN_packed, img_out, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
}

void morpho_compute_opening_closing3(morpho_data_t *morpho_data, const uint8_t **img_in, uint8_t **img_out, const int i0,
                             const int i1, const int j0, const int j1)
{
    assert(img_in != NULL);
    assert(img_out != NULL);
    bitpack((const uint8_t **)img_in, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_erosion3_packed((const uint8_t **)morpho_data->IN_packed, morpho_data->IB_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_dilation3_packed((const uint8_t **)morpho_data->IB_packed, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_dilation3_packed((const uint8_t **)morpho_data->IN_packed, morpho_data->IB_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    morpho_compute_erosion3_packed((const uint8_t **)morpho_data->IB_packed, morpho_data->IN_packed, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
    bitunpack((const uint8_t **)morpho_data->IN_packed, img_out, i0, i1, j0, j1, morpho_data->carry, morpho_data->ncolp);
}
