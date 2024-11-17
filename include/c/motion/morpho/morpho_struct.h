/*!
 * \file
 * \brief Morphology structure.
 */

#pragma once
#include <stdint.h>

/**
 *  Inner data required to perform morphology.
 */
typedef struct {
    int i0; /**< First \f$y\f$ index in the image (included). */
    int i1; /**< Last \f$y\f$ index in the image (included). */
    int j0; /**< First \f$x\f$ index in the image (included). */
    int j1; /**< Last \f$x\f$ index in the image (included). */
    int carry; 
    int ncolp; 
    uint8_t **IB; /**< Temporary binary image. */
    uint8_t **IB_packed; /**< Temporary packed binary image. */
    uint8_t **IN_packed; /**< Temporary packed input binary image. */
} morpho_data_t;
