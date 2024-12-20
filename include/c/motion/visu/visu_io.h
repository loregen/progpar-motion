/*!
 * \file
 * \brief Visualization.
 */

#pragma once

#include "motion/visu/visu_struct.h"
#include "motion/tracking/tracking_struct.h"

/**
 * Allocation and initialization of the visualization module.
 * @param path Path to the video or images.
 * @param start Start frame number (first frame is frame 0).
 * @param n_ffmpeg_threads Number of threads used in FFMPEG to encode the video sequence (0 means FFMPEG will decide).
 * @param img_height Images height.
 * @param img_width Images width.
 * @param pixfmt_in Pixels format (grayscale or RGB) of the input.
 * @param pixfmt_out Pixels format (grayscale or RGB) of the output.
 * @param codec_type Select the API to use for video codec (`VCDC_FFMPEG_IO`).
 * @param draw_track_id If 1, draw the track id corresponding to the bounding box.
 * @param win_play Boolean, if 0 write into a file, if 1 play in a SDL window.
 * @param ffmpeg_debug Print the ffmpeg command line.
 * @param ffmpeg_out_extra_opts Pass extra arguments to ffmpeg (can be NULL).
 * @param buff_size Number of frames to buffer.
 * @param max_RoIs_size Max number of RoIs to allocate per frame.
 * @param Number of skipped frames between two 'visu_display' calls (generally this is 0).
 * @return The allocated data.
 */
visu_data_t* visu_alloc_init(const char* path, const size_t start, const size_t n_ffmpeg_threads,
                             const size_t img_height, const size_t img_width, const enum pixfmt_e pixfmt_in,
                             const enum pixfmt_e pixfmt_out, const enum video_codec_e codec_type,
                             const uint8_t draw_track_id, const int win_play, const uint8_t ffmpeg_debug,
                             const char* ffmpeg_out_extra_opts, const size_t buff_size, const size_t max_RoIs_size,
                             const uint8_t skip_fra);

/**
 * Display a frame. If the buffer is not fully filled: display nothing and just copy the current frame to the buffer.
 * @param visu A pointer of previously allocated inner visu data.
 * @param img Input grayscale/RGB image (2D array \f$[\texttt{img\_height}][\texttt{img\_width}]\f$).
 * @param RoIs Last RoIs to bufferize.
 * @param n_RoIs Number of connected-components (= number of RoIs) in the 2D array of `labels`.
 * @param tracks A vector of tracks.
 * @param frame_id the current frame id.
 */
void visu_display(visu_data_t* visu, const uint8_t** img, const RoI_t* RoIs, const size_t n_RoIs,
                  const vec_track_t tracks, const uint32_t frame_id);

/**
 * Display all the remaining frames (= flush the the buffer).
 * @param visu A pointer of previously allocated inner visu data.
 * @param tracks A vector of tracks.
 */
void visu_flush(visu_data_t* visu, const vec_track_t tracks);

/**
 * Deallocation of inner visu data.
 * @param video A pointer of video writer inner data.
 */
void visu_free(visu_data_t* visu);
