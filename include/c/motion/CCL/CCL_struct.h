/*!
 * \file
 * \brief Connected-Component Labeling (CCL) structures.
 */

#pragma once

#include <stdint.h>

/**
 *  Inner CCL data required to perform labeling (for Arthur HENNEQUIN's LSL implementation).
 */
typedef struct {
    int i0; /**< First \f$y\f$ index in the image (included). */
    int i1; /**< Last \f$y\f$ index in the image (included). */
    int j0; /**< First \f$x\f$ index in the image (included). */
    int j1; /**< Last \f$x\f$ index in the image (included). */
    uint32_t** er;  /**< Relative labels. */
    //uint32_t** ea;  // Absolute labels.
    uint32_t** era; /**< Relative <-> absolute labels equivalences. */
    uint32_t** rlc; /**< Run-length coding. */
    uint32_t* eq;   /**< Table of equivalence. */
    uint32_t* ner;  /**< Number of relative labels. */
} CCL_data_t;
