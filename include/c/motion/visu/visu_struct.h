/*!
 * \file
 * \brief Visualization structures.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#include "motion/video/video_struct.h"
#include "motion/image/image_struct.h"
#include "motion/features/features_struct.h"

/**
 *  Visualization structure.
 */
typedef struct {
    video_writer_t *video_writer; /*!< Video writer to encode the results in a file or show the result to the screen. */
    size_t img_height; /*!< Images height. */
    size_t img_width; /*!< Images width. */
    img_data_t *img_data; /*!< Proxy data to draw bounding boxes. */
    uint8_t ***I; /*!< Array of images (= buffer). */
    RoI_t **RoIs; /*!< Array of RoIs (= buffer). */
    uint32_t *frame_ids; /*!< RoIs corresponding frame ids. */
    size_t buff_size; /*!< Size of the bufferization. */
    size_t buff_id_read; /*!< Index of the current buffer to read. */
    size_t buff_id_write; /*!< Index of the current buffer to write. */
    size_t n_filled_buff; /*!< Number of filled buffers. */
    uint8_t draw_track_id; /*!< If 1, draw the track id corresponding to the bounding box. */
    uint8_t skip_fra; /*!< Number of skipped frames between two 'visu_display' calls (generally this is 0). */
    enum pixfmt_e pixfmt_in; /*!< Pixel format of the input images */

    vec_BB_t BBs;
    vec_color_e BBs_color;
} visu_data_t;
