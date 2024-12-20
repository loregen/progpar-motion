#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <nrc2.h>

#include "vec.h"
#include "motion/video/video_io.h"
#include "motion/visu/visu_io.h"
#include "motion/features/features_compute.h"
#include "motion/image/image_compute.h"

visu_data_t* visu_alloc_init(const char* path, const size_t start, const size_t n_ffmpeg_threads,
                             const size_t img_height, const size_t img_width, const enum pixfmt_e pixfmt_in,
                             const enum pixfmt_e pixfmt_out, const enum video_codec_e codec_type,
                             const uint8_t draw_track_id, const int win_play, const uint8_t ffmpeg_debug,
                             const char* ffmpeg_out_extra_opts, const size_t buff_size, const size_t max_RoIs_size,
                             const uint8_t skip_fra) {
    assert(buff_size > 0);
    visu_data_t* visu = (visu_data_t*)malloc(sizeof(visu_data_t));
    visu->img_height = img_height;
    visu->img_width = img_width;
    visu->video_writer = video_writer_alloc_init(path, start, n_ffmpeg_threads, visu->img_height, visu->img_width,
                                                 pixfmt_out, codec_type, win_play, ffmpeg_debug, ffmpeg_out_extra_opts);
    visu->buff_size = buff_size;
    visu->skip_fra = skip_fra;
    visu->buff_id_read = 0;
    visu->buff_id_write = 0;
    visu->n_filled_buff = 0;
    visu->I = (uint8_t***)malloc(visu->buff_size * sizeof(uint8_t**));
    visu->RoIs = (RoI_t**)malloc(visu->buff_size * sizeof(RoI_t**));
    visu->frame_ids = (uint32_t*)malloc(visu->buff_size * sizeof(uint32_t));
    visu->pixfmt_in = pixfmt_in;

    size_t pixsize = image_get_pixsize(visu->pixfmt_in);
    for (size_t i = 0; i < visu->buff_size; i++) {
        visu->I[i] = ui8matrix(0, visu->img_height + 1, 0, (visu->img_width + 1) * pixsize);
        visu->RoIs[i] = features_alloc_RoIs(max_RoIs_size);
    }
    visu->img_data = image_color_alloc(img_height, img_width);
    visu->BBs = (vec_BB_t)vector_create();
    visu->BBs_color = (vec_color_e)vector_create();
    visu->draw_track_id = draw_track_id;

    return visu;
}

void _add_to_BB_coord_list(vec_BB_t* BBs, vec_color_e* BBs_color, size_t elem, int rx, int ry, int bb_x, int bb_y,
                           int frame_id, int track_id, int is_extrapolated, enum color_e color) {
    size_t vs = vector_size(*BBs);
    BB_t* BB_elem = (vs == elem) ? vector_add_asg(BBs) : &(*BBs)[elem];
    BB_elem->frame_id = frame_id;
    BB_elem->track_id = track_id;
    BB_elem->bb_x = bb_x;
    BB_elem->bb_y = bb_y;
    BB_elem->rx = rx;
    BB_elem->ry = ry;
    BB_elem->is_extrapolated = is_extrapolated;

    if (vs == elem)
        vector_add(BBs_color, COLOR_MISC);
    enum color_e* BB_color_elem = &(*BBs_color)[elem];
    *BB_color_elem = color;
}

void _visu_write_or_play(visu_data_t* visu, const vec_track_t tracks) {
    const size_t real_buff_id_read = visu->buff_id_read % visu->buff_size;
    const size_t frame_id = visu->frame_ids[real_buff_id_read];
    int cpt = 0;
    size_t n_tracks = vector_size(tracks);
    for (size_t i = 0; i < n_tracks; i++) {
        const uint32_t track_id = tracks[i].id;
        if (track_id && (tracks[i].end.frame >= frame_id && tracks[i].begin.frame <= frame_id)) {
            const size_t offset = (tracks[i].end.frame - frame_id) / (visu->skip_fra + 1);
            assert(tracks[i].RoIs_id != NULL);
            const size_t RoIs_id_size = vector_size(tracks[i].RoIs_id);
            assert(RoIs_id_size > offset);
            const uint32_t RoI_id = tracks[i].RoIs_id[(RoIs_id_size - 1) - offset];

            RoI_t *RoIs_tmp = visu->RoIs[real_buff_id_read];
            if (RoI_id) {
                const uint32_t track_x = (uint32_t)roundf(RoIs_tmp[RoI_id -1].x);
                const uint32_t track_y = (uint32_t)roundf(RoIs_tmp[RoI_id -1].y);
                const uint32_t track_rx = (RoIs_tmp[RoI_id -1].xmax - RoIs_tmp[RoI_id -1].xmin) / 2;
                const uint32_t track_ry = (RoIs_tmp[RoI_id -1].ymax - RoIs_tmp[RoI_id -1].ymin) / 2;

                int track_is_extrapolated = 0;
                _add_to_BB_coord_list(&visu->BBs, &visu->BBs_color, cpt, track_rx, track_ry, track_x, track_y, frame_id,
                                      track_id, track_is_extrapolated, COLOR_GREEN);
                cpt++;
           }
        }
    }

    const int is_gt_path = 0;
    image_color_draw_BBs(visu->img_data, (const uint8_t**)visu->I[real_buff_id_read], visu->pixfmt_in,
                         (const BB_t*)visu->BBs, (const enum color_e*)visu->BBs_color, cpt, visu->draw_track_id,
                         is_gt_path);

#ifdef MOTION_OPENCV_LINK
    image_color_draw_frame_id(visu->img_data, frame_id);
#endif

    video_writer_save_frame(visu->video_writer, (const uint8_t**)image_color_get_pixels_2d(visu->img_data));
}

void visu_display(visu_data_t* visu, const uint8_t** img, const RoI_t* RoIs, const size_t n_RoIs,
                  const vec_track_t tracks, const uint32_t frame_id) {
    // ------------------------
    // write or play image ----
    // ------------------------
    if (visu->n_filled_buff == visu->buff_size) {
        _visu_write_or_play(visu, tracks);

        visu->n_filled_buff--;
        visu->buff_id_read++;
    }

    // ------------------------
    // bufferize frame --------
    // ------------------------
    assert(visu->n_filled_buff <= visu->buff_size);

    size_t pixsize = image_get_pixsize(visu->pixfmt_in);
    const size_t real_buff_id_write = visu->buff_id_write % visu->buff_size;
    for (size_t i = 0; i < visu->img_height; i++)
        memcpy(visu->I[real_buff_id_write][i], img[i], visu->img_width * sizeof(uint8_t) * pixsize);
    memcpy(visu->RoIs[real_buff_id_write], RoIs, n_RoIs * sizeof(RoI_t));
    visu->frame_ids[real_buff_id_write] = frame_id;

    visu->n_filled_buff++;
    visu->buff_id_write++;
}

void visu_flush(visu_data_t* visu, const vec_track_t tracks) {
    while (visu->n_filled_buff) {
        _visu_write_or_play(visu, tracks);

        visu->n_filled_buff--;
        visu->buff_id_read++;
    };
}

void visu_free(visu_data_t* visu) {
    size_t pixsize = image_get_pixsize(visu->pixfmt_in);
    video_writer_free(visu->video_writer);
    for (size_t i = 0; i < visu->buff_size; i++) {
        free_ui8matrix(visu->I[i], 0, visu->img_height + 1, 0, (visu->img_width + 1) * pixsize);
        features_free_RoIs(visu->RoIs[i]);
    }
    free(visu->I);
    free(visu->RoIs);
    free(visu->frame_ids);
    image_color_free(visu->img_data);
    vector_free(visu->BBs);
    vector_free(visu->BBs_color);
    free(visu);
}
