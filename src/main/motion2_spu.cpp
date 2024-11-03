#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <nrc2.h>
#include <math.h>
#include <streampu.hpp>

#include "vec.h"

#include "motion/args.h"
#include "motion/tools.h"
#include "motion/macros.h"

#include "motion/CCL.h"
#include "motion/features.h"
#include "motion/kNN.h"
#include "motion/tracking.h"
#include "motion/video.h"
#include "motion/image.h"
#include "motion/video.h"
#include "motion/sigma_delta.h"
#include "motion/morpho.h"
#include "motion/visu.h"

#include "motion/wrapper/Video_reader.hpp"
#include "motion/wrapper/Logger_frame.hpp"
#include "motion/wrapper/Logger_RoIs.hpp"
#include "motion/wrapper/Logger_kNN.hpp"
#include "motion/wrapper/Logger_tracks.hpp"
#include "motion/wrapper/Visu.hpp"
#include "motion/wrapper/Sigma_delta.hpp"
#include "motion/wrapper/Morpho.hpp"
#include "motion/wrapper/CCL.hpp"
#include "motion/wrapper/Features_CCA.hpp"
#include "motion/wrapper/Features_filter.hpp"
#include "motion/wrapper/KNN.hpp"
#include "motion/wrapper/Tracking.hpp"


int main(int argc, char** argv) {

    // ---------------------------------- //
    // -- DEFAULT VALUES OF PARAMETERS -- //
    // ---------------------------------- //

    char* def_p_vid_in_path = NULL;
    int def_p_vid_in_start = 0;
    int def_p_vid_in_stop = 0;
    int def_p_vid_in_skip = 0;
    int def_p_vid_in_loop = 1;
    int def_p_vid_in_threads = 0;
    char def_p_vid_in_dec_hw[16] = "NONE";
    int def_p_sd_n = 2;
    char* def_p_ccl_fra_path = NULL;
    int def_p_flt_s_min = 50;
    int def_p_flt_s_max = 100000;
    int def_p_knn_k = 3;
    int def_p_knn_d = 10;
    float def_p_knn_s = 0.125f;
    int def_p_trk_ext_d = 5;
    int def_p_trk_ext_o = 3;
    int def_p_trk_obj_min = 2;
    char* def_p_trk_roi_path = NULL;
    char* def_p_log_path = NULL;
    int def_p_cca_roi_max1 = 65536; // Maximum number of RoIs
    int def_p_cca_roi_max2 = 8192; // Maximum number of RoIs after filtering
    char* def_p_vid_out_path = NULL;

    // ------------------------ //
    // -- CMD LINE ARGS HELP -- //
    // ------------------------ //

    if (args_find(argc, argv, "--help,-h")) {
        fprintf(stderr,
                "  --vid-in-path     Path to video file or to an images sequence                            [%s]\n",
                def_p_vid_in_path ? def_p_vid_in_path : "NULL");
        fprintf(stderr,
                "  --vid-in-start    Start frame id (included) in the video                                 [%d]\n",
                def_p_vid_in_start);
        fprintf(stderr,
                "  --vid-in-stop     Stop frame id (included) in the video (if set to 0, read entire video) [%d]\n",
                def_p_vid_in_stop);
        fprintf(stderr,
                "  --vid-in-skip     Number of frames to skip                                               [%d]\n",
                def_p_vid_in_skip);
        fprintf(stderr,
                "  --vid-in-buff     Bufferize all the video in global memory before executing the chain        \n");
        fprintf(stderr,
                "  --vid-in-loop     Number of times the video is read in loop                              [%d]\n",
                def_p_vid_in_loop);
        fprintf(stderr,
                "  --vid-in-threads  Select the number of threads to use to decode video input (in ffmpeg)  [%d]\n",
                def_p_vid_in_threads);
        fprintf(stderr,
                "  --vid-in-dec-hw   Select video decoder hardware acceleration ('NONE', 'NVDEC', 'VIDTB')  [%s]\n",
                def_p_vid_in_dec_hw);
        fprintf(stderr,
                "  --sd-n            Value of the N parameter in the Sigma-Delta algorithm                  [%d]\n",
                def_p_sd_n);
        fprintf(stderr,
                "  --ccl-fra-path    Path of the files for CC debug frames                                  [%s]\n",
                def_p_ccl_fra_path ? def_p_ccl_fra_path : "NULL");
#ifdef MOTION_OPENCV_LINK
        fprintf(stderr,
                "  --ccl-fra-id      Show the RoI/CC ids on the ouptut CC frames                                \n");
#endif
        fprintf(stderr,
                "  --cca-roi-max1    Maximum number of RoIs after CCA                                       [%d]\n",
                def_p_cca_roi_max1);
        fprintf(stderr,
                "  --cca-roi-max2    Maximum number of RoIs after surface filtering                         [%d]\n",
                def_p_cca_roi_max2);
        fprintf(stderr,
                "  --flt-s-min       Minimum surface of the CCs in pixels                                   [%d]\n",
                def_p_flt_s_min);
        fprintf(stderr,
                "  --flt-s-max       Maxumum surface of the CCs in pixels                                   [%d]\n",
                def_p_flt_s_max);
        fprintf(stderr,
                "  --knn-k           Maximum number of neighbors considered in k-NN algorithm               [%d]\n",
                def_p_knn_k);
        fprintf(stderr,
                "  --knn-d           Maximum distance in pixels between two images (in k-NN)                [%d]\n",
                def_p_knn_d);
        fprintf(stderr,
                "  --knn-s           Minimum surface ratio to match two CCs in k-NN                         [%f]\n",
                def_p_knn_s);
        fprintf(stderr,
                "  --trk-ext-d       Search radius in pixels for CC extrapolation (piece-wise tracking)     [%d]\n",
                def_p_trk_ext_d);
        fprintf(stderr,
                "  --trk-ext-o       Maximum number of frames to extrapolate (linear) for lost objects      [%d]\n",
                def_p_trk_ext_o);
        fprintf(stderr,
                "  --trk-obj-min     Minimum number of frames required to track an object                   [%d]\n",
                def_p_trk_obj_min);
        fprintf(stderr,
                "  --trk-roi-path    Path to the file containing the RoI ids for each track                 [%s]\n",
                def_p_trk_roi_path ? def_p_trk_roi_path : "NULL");
        fprintf(stderr,
                "  --log-path        Path of the output statistics, only required for debugging purpose     [%s]\n",
                def_p_log_path ? def_p_log_path : "NULL");
        fprintf(stderr,
                "  --vid-out-path    Path to video file or to an images sequence to write the output        [%s]\n",
                def_p_vid_out_path ? def_p_vid_out_path : "NULL");
        fprintf(stderr,
                "  --vid-out-play    Show the output video in a SDL window                                      \n");
#ifdef MOTION_OPENCV_LINK
        fprintf(stderr,
                "  --vid-out-id      Draw the track ids on the ouptut video                                     \n");
#endif
        fprintf(stderr,
                "  --stats           Show the average latency of each task                                      \n");
        fprintf(stderr,
                "  --help, -h        This help                                                                  \n");
        exit(1);
    }

    // ------------------------- //
    // -- PARSE CMD LINE ARGS -- //
    // ------------------------- //

    const char* p_vid_in_path = args_find_char(argc, argv, "--vid-in-path", def_p_vid_in_path);
    const int p_vid_in_start = args_find_int_min(argc, argv, "--vid-in-start", def_p_vid_in_start, 0);
    const int p_vid_in_stop = args_find_int_min(argc, argv, "--vid-in-stop", def_p_vid_in_stop, 0);
    const int p_vid_in_skip = args_find_int_min(argc, argv, "--vid-in-skip", def_p_vid_in_skip, 0);
    const int p_vid_in_buff = args_find(argc, argv, "--vid-in-buff");
    const int p_vid_in_loop = args_find_int_min(argc, argv, "--vid-in-loop", def_p_vid_in_loop, 1);
    const int p_vid_in_threads = args_find_int_min(argc, argv, "--vid-in-threads", def_p_vid_in_threads, 0);
    const char* p_vid_in_dec_hw = args_find_char(argc, argv, "--vid-in-dec-hw", def_p_vid_in_dec_hw);
    const int p_sd_n = args_find_int_min(argc, argv, "--sd-n", def_p_sd_n, 0);
    const char* p_ccl_fra_path = args_find_char(argc, argv, "--ccl-fra-path", def_p_ccl_fra_path);
#ifdef MOTION_OPENCV_LINK
    const int p_ccl_fra_id = args_find(argc, argv, "--ccl-fra-id,--show-id");
#else
    const int p_ccl_fra_id = 0;
#endif
    const int p_cca_roi_max1 = args_find_int_min(argc, argv, "--cca-roi-max1", def_p_cca_roi_max1, 0);
    const int p_cca_roi_max2 = args_find_int_min(argc, argv, "--cca-roi-max2", def_p_cca_roi_max2, 0);
    const int p_flt_s_min = args_find_int_min(argc, argv, "--flt-s-min", def_p_flt_s_min, 0);
    const int p_flt_s_max = args_find_int_min(argc, argv, "--flt-s-max", def_p_flt_s_max, 0);
    const int p_knn_k = args_find_int_min(argc, argv, "--knn-k", def_p_knn_k, 0);
    const int p_knn_d = args_find_int_min(argc, argv, "--knn-d", def_p_knn_d, 0);
    const float p_knn_s = args_find_float_min_max(argc, argv, "--knn-s", def_p_knn_s, 0.f, 1.f);
    const int p_trk_ext_d = args_find_int_min(argc, argv, "--trk-ext-d", def_p_trk_ext_d, 0);
    const int p_trk_ext_o = args_find_int_min_max(argc, argv, "--trk-ext-o", def_p_trk_ext_o, 0, 255);
    const int p_trk_obj_min = args_find_int_min(argc, argv, "--trk-obj-min", def_p_trk_obj_min, 2);
    const char* p_trk_roi_path = args_find_char(argc, argv, "--trk-roi-path", def_p_trk_roi_path);
    const char* p_log_path = args_find_char(argc, argv, "--log-path", def_p_log_path);
    const char* p_vid_out_path = args_find_char(argc, argv, "--vid-out-path", def_p_vid_out_path);
    const int p_vid_out_play = args_find(argc, argv, "--vid-out-play");
#ifdef MOTION_OPENCV_LINK
    const int p_vid_out_id = args_find(argc, argv, "--vid-out-id");
#else
    const int p_vid_out_id = 0;
#endif
    const int p_stats = args_find(argc, argv, "--stats");

    // --------------------- //
    // -- HEADING DISPLAY -- //
    // --------------------- //

    printf("#  --------------- \n");
    printf("# |  MOTION2 SPU  |\n");
    printf("#  --------------- \n");
    printf("#\n");
    printf("# Parameters:\n");
    printf("# -----------\n");
    printf("#  * vid-in-path    = %s\n", p_vid_in_path);
    printf("#  * vid-in-start   = %d\n", p_vid_in_start);
    printf("#  * vid-in-stop    = %d\n", p_vid_in_stop);
    printf("#  * vid-in-skip    = %d\n", p_vid_in_skip);
    printf("#  * vid-in-buff    = %d\n", p_vid_in_buff);
    printf("#  * vid-in-loop    = %d\n", p_vid_in_loop);
    printf("#  * vid-in-threads = %d\n", p_vid_in_threads);
    printf("#  * vid-in-dec-hw  = %s\n", p_vid_in_dec_hw);
    printf("#  * sd-n           = %d\n", p_sd_n);
    printf("#  * ccl-fra-path   = %s\n", p_ccl_fra_path);
#ifdef MOTION_OPENCV_LINK
    printf("#  * ccl-fra-id     = %d\n", p_ccl_fra_id);
#endif
    printf("#  * cca-roi-max1   = %d\n", p_cca_roi_max1);
    printf("#  * cca-roi-max2   = %d\n", p_cca_roi_max2);
    printf("#  * flt-s-min      = %d\n", p_flt_s_min);
    printf("#  * flt-s-max      = %d\n", p_flt_s_max);
    printf("#  * knn-k          = %d\n", p_knn_k);
    printf("#  * knn-d          = %d\n", p_knn_d);
    printf("#  * knn-s          = %1.3f\n", p_knn_s);
    printf("#  * trk-ext-d      = %d\n", p_trk_ext_d);
    printf("#  * trk-ext-o      = %d\n", p_trk_ext_o);
    printf("#  * trk-obj-min    = %d\n", p_trk_obj_min);
    printf("#  * trk-roi-path   = %s\n", p_trk_roi_path);
    printf("#  * log-path       = %s\n", p_log_path);
    printf("#  * vid-out-path   = %s\n", p_vid_out_path);
    printf("#  * vid-out-play   = %d\n", p_vid_out_play);
#ifdef MOTION_OPENCV_LINK
    printf("#  * vid-out-id     = %d\n", p_vid_out_id);
#endif
    printf("#  * stats          = %d\n", p_stats);

    printf("#\n");

    // -------------------------- //
    // -- CMD LINE ARGS CHECKS -- //
    // -------------------------- //

    if (!p_vid_in_path) {
        fprintf(stderr, "(EE) '--vid-in-path' is missing\n");
        exit(1);
    }
    if (p_vid_in_stop && p_vid_in_stop < p_vid_in_start) {
        fprintf(stderr, "(EE) '--vid-in-stop' has to be higher than '--vid-in-start'\n");
        exit(1);
    }
#ifdef MOTION_OPENCV_LINK
    if (p_ccl_fra_id && !p_ccl_fra_path)
        fprintf(stderr, "(WW) '--ccl-fra-id' has to be combined with the '--ccl-fra-path' parameter\n");
#endif
    if (p_vid_out_path && p_vid_out_play)
        fprintf(stderr, "(WW) '--vid-out-path' will be ignore because '--vid-out-play' is set\n");
#ifdef MOTION_OPENCV_LINK
    if (p_vid_out_id && !p_vid_out_path && !p_vid_out_play)
        fprintf(stderr,
                "(WW) '--vid-out-id' will be ignore because neither '--vid-out-play' nor 'p_vid_out_path' are set\n");
#endif

    // --------------------------------------- //
    // -- VIDEO ALLOCATION & INITIALISATION -- //
    // --------------------------------------- //

    TIME_POINT(start_alloc_init);
    Video_reader video(p_vid_in_path, p_vid_in_start, p_vid_in_stop, p_vid_in_skip,
                       p_vid_in_buff, p_vid_in_threads, VCDC_FFMPEG_IO, video_hwaccel_str_to_enum(p_vid_in_dec_hw));
    int i0 = video.get_i0(), i1 = video.get_i1(), j0 = video.get_j0(), j1 = video.get_j1();
    video.set_loop_size(p_vid_in_loop);

    std::unique_ptr<Logger_frame> log_fra;
    if (p_ccl_fra_path)
        log_fra.reset(new Logger_frame(p_ccl_fra_path, p_vid_in_start, p_ccl_fra_id, i0, i1, j0, j1, p_cca_roi_max2));

    // --------------------- //
    // -- DATA ALLOCATION -- //
    // --------------------- //

    // sigma_delta_data_t* sd_data0 = sigma_delta_alloc_data(i0, i1, j0, j1, 1, 254);
    // sigma_delta_data_t* sd_data1 = sigma_delta_alloc_data(i0, i1, j0, j1, 1, 254);
    // morpho_data_t* morpho_data0 = morpho_alloc_data(i0, i1, j0, j1);
    // morpho_data_t* morpho_data1 = morpho_alloc_data(i0, i1, j0, j1);
    RoI_t* RoIs_tmp0 = features_alloc_RoIs(p_cca_roi_max1);
    // RoI_t* RoIs0 = features_alloc_RoIs(p_cca_roi_max2);
    RoI_t* RoIs_tmp1 = features_alloc_RoIs(p_cca_roi_max1);
    // RoI_t* RoIs1 = features_alloc_RoIs(p_cca_roi_max2);
    // CCL_data_t* ccl_data0 = CCL_LSL_alloc_data(i0, i1, j0, j1);
    // CCL_data_t* ccl_data1 = CCL_LSL_alloc_data(i0, i1, j0, j1);
    kNN_data_t* knn_data = kNN_alloc_data(p_cca_roi_max2);
    tracking_data_t* tracking_data = tracking_alloc_data(MAX(p_trk_obj_min, p_trk_ext_o) + 1, p_cca_roi_max2);
    uint8_t **IG0 = ui8matrix(i0, i1, j0, j1); // grayscale input image at t - 1
    uint8_t **IG1 = ui8matrix(i0, i1, j0, j1); // grayscale input image at t
    uint8_t **IB0 = ui8matrix(i0, i1, j0, j1); // binary image (after Sigma-Delta) at t - 1
    uint8_t **IB1 = ui8matrix(i0, i1, j0, j1); // binary image (after Sigma-Delta) at t
    uint32_t **L10 = ui32matrix(i0, i1, j0, j1); // labels (CCL) at t - 1
    uint32_t **L11 = ui32matrix(i0, i1, j0, j1); // labels (CCL) at t
    uint32_t **L20 = NULL; // labels (CCL + surface filter) at t - 1
    uint32_t **L21 = NULL; // labels (CCL + surface filter) at t
    if (p_ccl_fra_path) {
        L20 = ui32matrix(i0, i1, j0, j1);
        L21 = ui32matrix(i0, i1, j0, j1);
    }

    Logger_RoIs log_RoIs(p_log_path ? p_log_path : "", p_vid_in_start, p_vid_in_skip, p_cca_roi_max2, tracking_data);
    Logger_kNN log_kNN(p_log_path ? p_log_path : "", p_vid_in_start, p_cca_roi_max2);
    Logger_tracks log_trk(p_log_path ? p_log_path : "", p_vid_in_start, tracking_data);

    // Processing modules allocation
    Sigma_delta sd0(i0, i1, j0, j1, p_sd_n), sd1(i0, i1, j0, j1, p_sd_n);
    Morpho morpho0(i0, i1, j0, j1), morpho1(i0, i1, j0, j1);
    CCL ccl0(i0, i1, j0, j1, 0), ccl1(i0, i1, j0, j1, 0);
    CCA cca0(i0, i1, j0, j1, p_cca_roi_max1), cca1(i0, i1, j0, j1, p_cca_roi_max1);
    Features_filter features0(i0, i1, j0, j1, p_flt_s_max, p_flt_s_min, p_cca_roi_max1, p_cca_roi_max2),features1(i0, i1, j0, j1, p_flt_s_max, p_flt_s_min, p_cca_roi_max1, p_cca_roi_max2);

    kNN knn(knn_data, p_knn_k, p_knn_d, p_knn_s, p_cca_roi_max2);

    std::unique_ptr<Visu> visu;

    Tracking tracking(tracking_data, p_trk_ext_d, p_trk_obj_min, p_trk_roi_path != NULL || visu,
                          p_trk_ext_o, p_knn_s, p_cca_roi_max2);

    if (p_vid_out_play || p_vid_out_path) {
        const uint8_t n_threads = 1;
        visu.reset(new Visu(p_vid_out_path, p_vid_in_start, n_threads, i0, i1, j0, j1, PIXFMT_GRAY8, PIXFMT_RGB24,
                            VCDC_FFMPEG_IO, p_vid_out_id, p_vid_out_play, p_trk_obj_min, p_cca_roi_max2, p_vid_in_skip,
                            tracking_data));
    }



    // ------------------------- //
    // -- DATA INITIALISATION -- //
    // ------------------------- //

    uint32_t cur_fra;
    video["generate::out_img_gray8"].bind(&IG1[0][0]);
    video["generate::out_frame"].bind(&cur_fra);
    video("generate").exec();

    // sigma_delta_init_data(sd_data0, (const uint8_t**)IG1, i0, i1, j0, j1);
    //sigma_delta_init_data(sd_data1, (const uint8_t**)IG1, i0, i1, j0, j1);
    sd0.init_data((const uint8_t**)IG1), sd1.init_data((const uint8_t**)IG1);

    zero_ui8matrix(IG0, i0, i1, j0, j1);
    zero_ui8matrix(IG1, i0, i1, j0, j1);
    zero_ui8matrix(IB0, i0, i1, j0, j1);
    zero_ui8matrix(IB1, i0, i1, j0, j1);
    zero_ui32matrix(L10, i0, i1, j0, j1);
    zero_ui32matrix(L11, i0, i1, j0, j1);
    if (p_ccl_fra_path) {
        zero_ui32matrix(L20, i0, i1, j0, j1);
        zero_ui32matrix(L21, i0, i1, j0, j1);
    }
    // morpho_init_data(morpho_data0);
    // morpho_init_data(morpho_data1);
    // CCL_LSL_init_data(ccl_data0);
    // CCL_LSL_init_data(ccl_data1);
    features_init_RoIs(RoIs_tmp0, p_cca_roi_max1);
    features_init_RoIs(RoIs_tmp1, p_cca_roi_max1);
    // features_init_RoIs(RoIs0, p_cca_roi_max2);
    // features_init_RoIs(RoIs1, p_cca_roi_max2);
    kNN_init_data(knn_data);
    tracking_init_data(tracking_data);

    if (visu) {
        uint32_t n_RoIs1 = 0;
        (*visu)["display::in_frame"].bind(&cur_fra);
        (*visu)["display::in_img"].bind(IG1[0]);
        (*visu)["display::in_RoIs"] = knn["match::out_RoIs1"];
        (*visu)["display::in_n_RoIs"] = features1["filter::out_n_RoIs"];
        (*visu)("display").exec();
    }

    TIME_POINT(stop_alloc_init);
    printf("# Allocations and initialisations took %6.3f sec\n", TIME_ELAPSED2_SEC(start_alloc_init, stop_alloc_init));

    // --------------------- //
    // -- PROCESSING LOOP -- //
    // --------------------- //

    printf("# The program is running...\n");
    size_t n_moving_objs = 0, n_processed_frames = 0;
    TIME_SETA(dec_a); TIME_SETA(sd_a); TIME_SETA(mrp_a); TIME_SETA(ccl_a); TIME_SETA(cca_a); TIME_SETA(flt_a);
    TIME_SETA(knn_a); TIME_SETA(trk_a); TIME_SETA(log_a); TIME_SETA(vis_a);
    TIME_POINT(start_compute);
    while (1) {
        // step 0: video decoding
        TIME_POINT(dec_b);
        try {
            video("generate").exec();
        } catch (const spu::tools::processing_aborted&) {}
        TIME_POINT(dec_e);
        TIME_ACC(dec_a, dec_b, dec_e);

        // loop stop condition (= end of the video)
        if (video.is_done())
            break;

        fprintf(stderr, "(II) Frame n°%4d", cur_fra);

        // -------------------------------------- //
        // -- IMAGE PROCESSING CHAIN EXECUTION -- //
        // -------------------------------------- //

        // ------------------------- //
        // -- Processing at t - 1 -- //
        // ------------------------- //

        //uint32_t n_RoIs0 = 0;
        if (n_processed_frames > 0) {
            // step 1: motion detection (per pixel) with Sigma-Delta algorithm
            TIME_POINT(sd_b);
            sd0["compute::in_img"].bind(IG0[0]);
            sd0("compute").exec();
            TIME_POINT(sd_e);
            TIME_ACC(sd_a, sd_b, sd_e);

            // step 2: mathematical morphology
            TIME_POINT(mrp_b);
            morpho0["compute::in_img"] = sd0["compute::out_img"];
            morpho0("compute").exec();
            TIME_POINT(mrp_e);
            TIME_ACC(mrp_a, mrp_b, mrp_e);

            // step 3: connected components labeling (CCL)
            TIME_POINT(ccl_b);
            uint32_t n_RoIs_tmp0 = 0;
            ccl0["apply::in_img"] = morpho0["compute::out_img"];
            //ccl0["apply::out_n_RoIs"].bind(&n_RoIs_tmp0);
            ccl0("apply").exec();
            //assert(n_RoIs_tmp0 <= (uint32_t)def_p_cca_roi_max1);

            TIME_POINT(ccl_e);
            TIME_ACC(ccl_a, ccl_b, ccl_e);

            // step 4: connected components analysis (CCA): from image of labels to "regions of interest" (RoIs)
            TIME_POINT(cca_b);
            cca0["extract::in_labels"]= ccl0["apply::out_labels"];
            cca0["extract::in_n_RoIs"]= ccl0["apply::out_n_RoIs"];
            cca0("extract").exec();
            TIME_POINT(cca_e);
            TIME_ACC(cca_a, cca_b, cca_e);

            // step 5: surface filtering (rm too small and too big RoIs)
            TIME_POINT(flt_b);
            features0["filter::in_labels"] = ccl0["apply::out_labels"];
            features0["filter::in_n_RoIs"] = ccl0["apply::out_n_RoIs"];
            features0["filter::in_RoIs"]   = cca0["extract::out_RoIs"];
            //features0["filter::out_RoIs"].bind((uint8_t*)RoIs0);
            //features0["filter::out_n_RoIs"].bind(&n_RoIs0);
            features0("filter").exec();
            TIME_POINT(flt_e);
            TIME_ACC(flt_a, flt_b, flt_e);
        }

        // --------------------- //
        // -- Processing at t -- //
        // --------------------- //

        // step 1: motion detection (per pixel) with Sigma-Delta algorithm
        TIME_POINT(sd_b);
        sd1["compute::in_img"].bind(IG1[0]);
        sd1("compute").exec();
        TIME_POINT(sd_e);
        TIME_ACC(sd_a, sd_b, sd_e);

        // step 2: mathematical morphology
        TIME_POINT(mrp_b);
        morpho1["compute::in_img"]= sd1["compute::out_img"];
        morpho1("compute").exec();
        TIME_POINT(mrp_e);
        TIME_ACC(mrp_a, mrp_b, mrp_e);

        // step 3: connected components labeling (CCL)
        TIME_POINT(ccl_b);
        uint32_t n_RoIs_tmp1;
        ccl1["apply::in_img"] = morpho1["compute::out_img"];
        //ccl1["apply::out_n_RoIs"].bind(&n_RoIs_tmp1);
        ccl1("apply").exec();
        //assert(n_RoIs_tmp1 <= (uint32_t)def_p_cca_roi_max1);
        TIME_POINT(ccl_e);
        TIME_ACC(ccl_a, ccl_b, ccl_e);

        // step 4: connected components analysis (CCA): from image of labels to "regions of interest" (RoIs)
        TIME_POINT(cca_b);
        cca1["extract::in_labels"]= ccl1["apply::out_labels"];
        cca1["extract::in_n_RoIs"]= ccl1["apply::out_n_RoIs"];
        cca1("extract").exec();
        TIME_POINT(cca_e);
        TIME_ACC(cca_a, cca_b, cca_e);

        // step 5: surface filtering (rm too small and too big RoIs)
        TIME_POINT(flt_b);
        //uint32_t n_RoIs1;
        features1["filter::in_labels"] = ccl1["apply::out_labels"];
        features1["filter::in_n_RoIs"] = ccl1["apply::out_n_RoIs"];
        features1["filter::in_RoIs"]   = cca1["extract::out_RoIs"];
        //features1["filter::out_RoIs"].bind((uint8_t*)RoIs1);
        //features1["filter::out_n_RoIs"].bind(&n_RoIs1);
        features1("filter").exec();
        TIME_POINT(flt_e);
        TIME_ACC(flt_a, flt_b, flt_e);

        // ----------------------------- //
        // -- Associations (t - 1, t) -- //
        // ----------------------------- //

        // step 6: k-NN matching (RoIs associations)
        TIME_POINT(knn_b);
        // kNN knn(knn_data, n_RoIs0, n_RoIs1, p_knn_k, p_knn_d, p_knn_s);
        uint32_t n_assoc;
        //kNN_match(knn_data, RoIs0, n_RoIs0, RoIs1, n_RoIs1, p_knn_k, p_knn_d, p_knn_s);
        knn["match::in_RoIs0"] = features0["filter::out_RoIs"];
        knn["match::in_n_RoIs0"] = features0["filter::out_n_RoIs"];
        knn["match::in_RoIs1"] = features1["filter::out_RoIs"];
        knn["match::in_n_RoIs1"] = features1["filter::out_n_RoIs"];
        //knn["match::out_RoIs1"] = knn["match::in_RoIs1"];
        //knn["match::out_RoIs0"].bind((uint8_t*)RoIs0);
        //knn["match::out_RoIs1"].bind((uint8_t*)RoIs1);
        knn["match::out_n_assoc"].bind(&n_assoc);
        knn("match").exec();
        TIME_POINT(knn_e);
        TIME_ACC(knn_a, knn_b, knn_e);

        // step 7: temporal tracking
        TIME_POINT(trk_b);
        // tracking_perform(tracking_data, RoIs1, n_RoIs1, cur_fra, p_trk_ext_d, p_trk_obj_min,
        //                  p_trk_roi_path != NULL || visu, p_trk_ext_o, p_knn_s);
        //Tracking tracking(tracking_data, n_RoIs1, cur_fra, p_trk_ext_d, p_trk_obj_min, p_trk_roi_path != NULL || visu,
        //                  p_trk_ext_o, p_knn_s);
        //tracking["perform::in_RoIs"].bind((uint8_t*)RoIs1);
        tracking["perform::in_RoIs"] = knn["match::out_RoIs1"];
        tracking["perform::in_n_RoIs"] = features1["filter::out_n_RoIs"];
        tracking["perform::in_frame"] = video["generate::out_frame"];
        tracking("perform").exec();
        TIME_POINT(trk_e);
        TIME_ACC(trk_a, trk_b, trk_e);

        // ---------- //
        // -- LOGS -- //
        // ---------- //

        TIME_POINT(log_b);
        // save frames (CCs)
        if (p_ccl_fra_path) {
            (*log_fra)["write::in_labels"].bind(L21[0]);
            (*log_fra)["write::in_RoIs"] = knn["match::out_RoIs1"];
            (*log_fra)["write::in_n_RoIs"] = features1["filter::out_n_RoIs"];
            (*log_fra)("write").exec();
        }

        // save stats
        if (p_log_path) {
            log_RoIs["write::in_RoIs0"] = knn["match::out_RoIs0"];
            log_RoIs["write::in_n_RoIs0"] = features0["filter::out_n_RoIs"];
            log_RoIs["write::in_RoIs1"] = knn["match::out_RoIs1"];
            log_RoIs["write::in_n_RoIs1"] = features1["filter::out_n_RoIs"];
            log_RoIs["write::in_frame"].bind(&cur_fra);
            log_RoIs("write").exec();

            if (cur_fra > (uint32_t)p_vid_in_start) {
                log_kNN["write::in_nearest"].bind(knn_data->nearest[0]);
                log_kNN["write::in_distances"].bind(knn_data->distances[0]);
#ifdef MOTION_ENABLE_DEBUG
                log_kNN["write::in_conflicts"].bind(knn_data->conflicts);
#endif
                log_kNN["write::in_RoIs0"] = knn["match::out_RoIs0"];
                log_kNN["write::in_n_RoIs0"] = features0["filter::out_n_RoIs"];
                log_kNN["write::in_RoIs1"] = knn["match::out_RoIs1"];
                log_kNN["write::in_n_RoIs1"] = features1["filter::out_n_RoIs"];
                log_kNN["write::in_frame"].bind(&cur_fra);
                log_kNN("write").exec();

                log_trk["write::in_frame"].bind(&cur_fra);
                log_trk("write").exec();
            }
        }
        TIME_POINT(log_e);
        TIME_ACC(log_a, log_b, log_e);

        // display the result to the screen or write it into a video file
        TIME_POINT(vis_b);
        if (visu) {
            (*visu)["display::in_frame"].bind(&cur_fra);
            (*visu)["display::in_img"].bind(IG1[0]);
            (*visu)["display::in_RoIs"] = knn["match::out_RoIs1"];
            (*visu)["display::in_n_RoIs"] = features1["filter::out_n_RoIs"];
            (*visu)("display").exec();
        }
        TIME_POINT(vis_e);
        TIME_ACC(vis_a, vis_b, vis_e);

        // swap IG0 <-> IG1 for the next frame
        uint8_t** tmp = IG0;
        IG0 = IG1;
        IG1 = tmp;
        // here we need to rebind the IG1 because we swapped the IG0 & IG1 buffers!
        video["generate::out_img_gray8"].bind(&IG1[0][0]);

        n_processed_frames++;
        n_moving_objs = tracking_count_objects(tracking_data->tracks);

        TIME_POINT(stop_compute);
        fprintf(stderr, " -- Time = %6.3f sec", TIME_ELAPSED2_SEC(start_compute, stop_compute));
        fprintf(stderr, " -- FPS = %4d", (int)(n_processed_frames / (TIME_ELAPSED2_SEC(start_compute, stop_compute))));
        fprintf(stderr, " -- Tracks = %3lu\r", (unsigned long)n_moving_objs);
        fflush(stderr);
    }
    TIME_POINT(stop_compute);
    fprintf(stderr, "\n");

    if (p_trk_roi_path) {
        FILE* f = fopen(p_trk_roi_path, "w");
        if (f == NULL) {
            fprintf(stderr, "(EE) error while opening '%s'\n", p_trk_roi_path);
            exit(1);
        }
        tracking_tracks_RoIs_id_write(f, tracking_data->tracks);
        fclose(f);
    }
    tracking_tracks_write(stdout, tracking_data->tracks);

    printf("# Tracks statistics:\n");
    printf("# -> Processed frames = %4u\n", (unsigned)n_processed_frames);
    printf("# -> Detected tracks  = %4lu\n", (unsigned long)n_moving_objs);
    printf("# -> Took %6.3f seconds (avg %d FPS)\n", TIME_ELAPSED2_SEC(start_compute, stop_compute),
           (int)(n_processed_frames / (TIME_ELAPSED2_SEC(start_compute, stop_compute))));
    if (p_stats) {
        printf("#\n");
        printf("# Average latencies: \n");
        printf("# -> Video decoding = %8.3f ms\n", TIME_ELAPSED_MS(dec_a) / n_processed_frames);
        printf("# -> Sigma-Delta    = %8.3f ms\n", TIME_ELAPSED_MS(sd_a)  / n_processed_frames);
        printf("# -> Morphology     = %8.3f ms\n", TIME_ELAPSED_MS(mrp_a) / n_processed_frames);
        printf("# -> CC Labeling    = %8.3f ms\n", TIME_ELAPSED_MS(ccl_a) / n_processed_frames);
        printf("# -> CC Analysis    = %8.3f ms\n", TIME_ELAPSED_MS(cca_a) / n_processed_frames);
        printf("# -> Filtering      = %8.3f ms\n", TIME_ELAPSED_MS(flt_a) / n_processed_frames);
        printf("# -> k-NN           = %8.3f ms\n", TIME_ELAPSED_MS(knn_a) / n_processed_frames);
        printf("# -> Tracking       = %8.3f ms\n", TIME_ELAPSED_MS(trk_a) / n_processed_frames);
        printf("# -> *Logs*         = %8.3f ms\n", TIME_ELAPSED_MS(log_a) / n_processed_frames);
        printf("# -> *Visu*         = %8.3f ms\n", TIME_ELAPSED_MS(vis_a) / n_processed_frames);
        TIME_SETA(total);
        TIME_ADD(total, dec_a); TIME_ADD(total,  sd_a); TIME_ADD(total, mrp_a); TIME_ADD(total, ccl_a);
        TIME_ADD(total, cca_a); TIME_ADD(total, flt_a); TIME_ADD(total, knn_a); TIME_ADD(total, trk_a);
        TIME_ADD(total, log_a); TIME_ADD(total, vis_a);
        double total = TIME_ELAPSED_MS(total) / n_processed_frames;
        printf("# => Total          = %8.3f ms [~%5.2f FPS]\n", total, 1000. / total);
    }

    // some frames have been buffered for the visualization, display or write these frames here
    if (visu)
        visu->flush();

    // ---------- //
    // -- FREE -- //
    // ---------- //

    /*sigma_delta_free_data(sd_data0);
    sigma_delta_free_data(sd_data1);
    morpho_free_data(morpho_data0);
    morpho_free_data(morpho_data1);*/
    free_ui8matrix(IG0, i0, i1, j0, j1);
    free_ui8matrix(IG1, i0, i1, j0, j1);
    free_ui8matrix(IB0, i0, i1, j0, j1);
    free_ui8matrix(IB1, i0, i1, j0, j1);
    free_ui32matrix(L10, i0, i1, j0, j1);
    free_ui32matrix(L11, i0, i1, j0, j1);
    if (p_ccl_fra_path) {
        free_ui32matrix(L20, i0, i1, j0, j1);
        free_ui32matrix(L21, i0, i1, j0, j1);
    }
    features_free_RoIs(RoIs_tmp0);
    features_free_RoIs(RoIs_tmp1);
    // features_free_RoIs(RoIs0);
    // features_free_RoIs(RoIs1);
    /*CCL_LSL_free_data(ccl_data0);
    CCL_LSL_free_data(ccl_data1);*/
    kNN_free_data(knn_data);
    tracking_free_data(tracking_data);

    printf("#\n");
    printf("# End of the program, exiting.\n");

    return EXIT_SUCCESS;
}
