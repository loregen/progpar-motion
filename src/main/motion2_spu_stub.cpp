/*
Name: Pierpaolo Marzo
Group: Lorenzo Gentile, Erik Fabrizzi
*/

/*
TASK 7

//Results on a i7 8750H compiled with gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0

##Dataflow model##

# -------------------------------------------||------------------------------||--------------------------------
#        Statistics for the given task       ||       Basic statistics       ||        Measured latency
#     ('*' = any, '-' = same as previous)    ||          on the task         ||
# -------------------------------------------||------------------------------||--------------------------------
# -------------|-------------------|---------||----------|----------|--------||----------|----------|----------
#       MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM
#              |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us)
# -------------|-------------------|---------||----------|----------|--------||----------|----------|----------
#  Sigma_delta |           compute |       * ||       21 |     0.41 |  22.61 || 19507.11 | 11758.35 | 32104.87
#  Sigma_delta |           compute |       * ||       20 |     0.39 |  21.78 || 19735.81 | 18381.48 | 21929.34
#       Morpho |           compute |       * ||       21 |     0.35 |  19.19 || 16562.93 | 16136.94 | 18453.12
#       Morpho |           compute |       * ||       20 |     0.33 |  18.43 || 16694.56 | 16070.32 | 18768.57
#          CCL |             apply |       * ||       21 |     0.07 |   4.11 ||  3546.07 |  2211.35 |  4312.85
#          CCL |             apply |       * ||       20 |     0.07 |   3.81 ||  3452.88 |  3045.27 |  4214.29
#          CCA |           extract |       * ||       21 |     0.05 |   2.62 ||  2260.15 |  1177.74 |  2859.60
#          CCA |           extract |       * ||       20 |     0.04 |   2.47 ||  2237.74 |  1888.54 |  2849.49
#        Video |          generate |       * ||       21 |     0.02 |   1.36 ||  1171.06 |     0.00 |  1654.24
# Features_filter |            filter |       * ||       21 |     0.02 |   1.15 ||   991.50 |   346.36 |  1779.69
# Features_filter |            filter |       * ||       20 |     0.02 |   1.07 ||   969.24 |   740.81 |  1575.58
#  Logger_RoIs |             write |       * ||       20 |     0.01 |   0.71 ||   646.49 |   157.91 |  2507.20
#      Delayer |          memorize |       * ||       20 |     0.00 |   0.28 ||   249.35 |   206.79 |   424.30
#      Delayer |           produce |       * ||       21 |     0.00 |   0.27 ||   233.53 |   167.76 |   481.54
# Logger_tracks |             write |       * ||       20 |     0.00 |   0.07 ||    62.23 |     8.59 |   269.59
#   Logger_kNN |             write |       * ||       20 |     0.00 |   0.05 ||    48.77 |    30.23 |   102.94
#          kNN |             match |       * ||       20 |     0.00 |   0.02 ||    14.56 |     5.56 |    24.17
#     Tracking |           perform |       * ||       20 |     0.00 |   0.01 ||     6.50 |     3.64 |    15.06
# -------------|-------------------|---------||----------|----------|--------||----------|----------|----------
#        TOTAL |                 * |       * ||       20 |     1.81 | 100.00 || 90604.12 | 73927.56 | 1.17e+05



##Straightforward version##

# Average latencies:
# -> Video decoding =    1.116 ms
# -> Sigma-Delta    =   36.653 ms
# -> Morphology     =   33.098 ms
# -> CC Labeling    =    6.566 ms
# -> CC Analysis    =    4.287 ms
# -> Filtering      =    0.012 ms
# -> k-NN           =    0.013 ms
# -> Tracking       =    0.004 ms
# -> Logs         =    0.458 ms
# -> Visu         =    0.000 ms
# => Total          =   82.209 ms [~12.16 FPS]

The difference in terms of statistics of the dataflow model and the straightforward version is negligible and depends on
the machine on which the code is executed and the compiler used.
In fact it results to be faster if compiled with clang and executed on a M1 chip, while it results to be slower in the shown setup.
This different behaviour may be due to the different optimization levels of the compilers and the different architectures of the CPUs,
which in some cases may lead to a better performance of the dataflow model and in other cases to a better performance of the straightforward version.
In general, the dataflow model should be faster than the straightforward version, because it allows to execute tasks in parallel and to exploit the
parallelism of the CPU, while the straightforward version executes tasks sequentially.
Though, in case of a small number of tasks, the overhead of the dataflow model may be higher than the benefits of the parallelism, leading to a slower execution.
*/

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

typedef std::tuple<std::vector<spu::runtime::Task *>,
                   std::vector<spu::runtime::Task *>,
                   std::vector<spu::runtime::Task *>>
    stage;

stage make_stage(std::initializer_list<spu::runtime::Task *> first_tasks,
                 std::initializer_list<spu::runtime::Task *> last_tasks,
                 std::initializer_list<spu::runtime::Task *> excluded_tasks)
{
        return std::make_tuple<std::vector<spu::runtime::Task *>,
                               std::vector<spu::runtime::Task *>,
                               std::vector<spu::runtime::Task *>>(
            first_tasks,
            last_tasks,
            excluded_tasks);
}

int main(int argc, char **argv)
{

        // ---------------------------------- //
        // -- DEFAULT VALUES OF PARAMETERS -- //
        // ---------------------------------- //

        char *def_p_vid_in_path = NULL;
        int def_p_vid_in_start = 0;
        int def_p_vid_in_stop = 0;
        int def_p_vid_in_skip = 0;
        int def_p_vid_in_loop = 1;
        int def_p_vid_in_threads = 0;
        char def_p_vid_in_dec_hw[16] = "NONE";
        int def_p_sd_n = 2;
        char *def_p_ccl_fra_path = NULL;
        int def_p_flt_s_min = 50;
        int def_p_flt_s_max = 100000;
        int def_p_knn_k = 3;
        int def_p_knn_d = 10;
        float def_p_knn_s = 0.125f;
        int def_p_trk_ext_d = 5;
        int def_p_trk_ext_o = 3;
        int def_p_trk_obj_min = 2;
        char *def_p_trk_roi_path = NULL;
        char *def_p_log_path = NULL;
        int def_p_cca_roi_max1 = 65536; // Maximum number of RoIs
        int def_p_cca_roi_max2 = 8192;  // Maximum number of RoIs after filtering
        char *def_p_vid_out_path = NULL;

        // ------------------------ //
        // -- CMD LINE ARGS HELP -- //
        // ------------------------ //

        if (args_find(argc, argv, "--help,-h"))
        {
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

        const char *p_vid_in_path = args_find_char(argc, argv, "--vid-in-path", def_p_vid_in_path);
        const int p_vid_in_start = args_find_int_min(argc, argv, "--vid-in-start", def_p_vid_in_start, 0);
        const int p_vid_in_stop = args_find_int_min(argc, argv, "--vid-in-stop", def_p_vid_in_stop, 0);
        const int p_vid_in_skip = args_find_int_min(argc, argv, "--vid-in-skip", def_p_vid_in_skip, 0);
        const int p_vid_in_buff = args_find(argc, argv, "--vid-in-buff");
        const int p_vid_in_loop = args_find_int_min(argc, argv, "--vid-in-loop", def_p_vid_in_loop, 1);
        const int p_vid_in_threads = args_find_int_min(argc, argv, "--vid-in-threads", def_p_vid_in_threads, 0);
        const char *p_vid_in_dec_hw = args_find_char(argc, argv, "--vid-in-dec-hw", def_p_vid_in_dec_hw);
        const int p_sd_n = args_find_int_min(argc, argv, "--sd-n", def_p_sd_n, 0);
        const char *p_ccl_fra_path = args_find_char(argc, argv, "--ccl-fra-path", def_p_ccl_fra_path);
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
        const char *p_trk_roi_path = args_find_char(argc, argv, "--trk-roi-path", def_p_trk_roi_path);
        const char *p_log_path = args_find_char(argc, argv, "--log-path", def_p_log_path);
        const char *p_vid_out_path = args_find_char(argc, argv, "--vid-out-path", def_p_vid_out_path);
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

        if (!p_vid_in_path)
        {
                fprintf(stderr, "(EE) '--vid-in-path' is missing\n");
                exit(1);
        }
        if (p_vid_in_stop && p_vid_in_stop < p_vid_in_start)
        {
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

        // TIME_POINT(start_alloc_init);
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

        kNN_data_t *knn_data = kNN_alloc_data(p_cca_roi_max2);
        tracking_data_t *tracking_data = tracking_alloc_data(MAX(p_trk_obj_min, p_trk_ext_o) + 1, p_cca_roi_max2);
        Logger_RoIs log_RoIs(p_log_path ? p_log_path : "", p_vid_in_start, p_vid_in_skip, p_cca_roi_max2, tracking_data);
        Logger_kNN log_kNN(p_log_path ? p_log_path : "", p_vid_in_start, p_cca_roi_max2);
        Logger_tracks log_trk(p_log_path ? p_log_path : "", p_vid_in_start, tracking_data);

        // Processing modules allocation
        Sigma_delta sd0(i0, i1, j0, j1, p_sd_n), sd1(i0, i1, j0, j1, p_sd_n);
        Morpho morpho0(i0, i1, j0, j1), morpho1(i0, i1, j0, j1);
        CCL ccl0(i0, i1, j0, j1, 0), ccl1(i0, i1, j0, j1, 0);
        CCA cca0(i0, i1, j0, j1, p_cca_roi_max1), cca1(i0, i1, j0, j1, p_cca_roi_max1);
        Features_filter features0(i0, i1, j0, j1, p_flt_s_max, p_flt_s_min, p_cca_roi_max1, p_cca_roi_max2), features1(i0, i1, j0, j1, p_flt_s_max, p_flt_s_min, p_cca_roi_max1, p_cca_roi_max2);

        kNN knn(knn_data, p_knn_k, p_knn_d, p_knn_s, p_cca_roi_max2);

        spu::module::Delayer<uint8_t> delayer(((i1 - i0) + 1) * ((j1 - j0) + 1), 0);

        std::unique_ptr<Visu> visu;

        if (p_vid_out_play || p_vid_out_path)
        {
                const uint8_t n_threads = 1;
                visu.reset(new Visu(p_vid_out_path, p_vid_in_start, n_threads, i0, i1, j0, j1, PIXFMT_GRAY8, PIXFMT_RGB24,
                                    VCDC_FFMPEG_IO, p_vid_out_id, p_vid_out_play, p_trk_obj_min, p_cca_roi_max2, p_vid_in_skip,
                                    tracking_data));
        }

        Tracking tracking(tracking_data, p_trk_ext_d, p_trk_obj_min, p_trk_roi_path != NULL || visu,
                          p_trk_ext_o, p_knn_s, p_cca_roi_max2);

        // ------------------------- //
        // -- DATA INITIALISATION -- //
        // ------------------------- //

        video("generate").exec();
        delayer.set_data(video["generate::out_img_gray8"].get_dataptr<uint8_t>());

        sd0.init_data((const uint8_t **)video["generate::out_img_gray8"].get_2d_dataptr<uint8_t>());
        sd1.init_data((const uint8_t **)video["generate::out_img_gray8"].get_2d_dataptr<uint8_t>());
        tracking_init_data(tracking_data);
        kNN_init_data(knn_data);

        // --------------------- //
        // -- PROCESSING LOOP -- //
        // --------------------- //

        printf("# The program is running...\n");
        size_t n_moving_objs = 0, n_processed_frames = 0;

        // -------------------------------------- //
        // -- IMAGE PROCESSING CHAIN EXECUTION -- //
        // -------------------------------------- //

        // ------------------------- //
        // -- Processing at t - 1 -- //
        // ------------------------- //

        // step 1: motion detection (per pixel) with Sigma-Delta algorithm
        sd0["compute::in_img"] = delayer["produce::out"];
        // step 2: mathematical morphology
        morpho0["computef::f_img"] = sd0["compute::out_img"];

        // step 3: connected components labeling (CCL)
        ccl0["apply::in_img"] = morpho0["computef::f_img"];
        // step 4: connected components analysis (CCA): from image of labels to "regions of interest" (RoIs)
        cca0["extract::in_labels"] = ccl0["apply::out_labels"];
        cca0["extract::in_n_RoIs"] = ccl0["apply::out_n_RoIs"];
        // step 5: surface filtering (rm too small and too big RoIs)
        features0["filterf::fwd_labels"] = ccl0["apply::out_labels"];
        features0["filterf::in_n_RoIs"] = ccl0["apply::out_n_RoIs"];
        features0["filterf::in_RoIs"] = cca0["extract::out_RoIs"];
        // --------------------- //
        // -- Processing at t -- //
        // --------------------- //

        // step 1: motion detection (per pixel) with Sigma-Delta algorithm
        sd1["compute::in_img"] = video["generate::out_img_gray8"];
        // step 2: mathematical morphology
        morpho1["computef::f_img"] = sd1["compute::out_img"];
        // step 3: connected components labeling (CCL)
        ccl1["apply::in_img"] = morpho1["computef::f_img"];
        // step 4: connected components analysis (CCA): from image of labels to "regions of interest" (RoIs)
        cca1["extract::in_labels"] = ccl1["apply::out_labels"];
        cca1["extract::in_n_RoIs"] = ccl1["apply::out_n_RoIs"];
        // step 5: surface filtering (rm too small and too big RoIs)
        features1["filterf::fwd_labels"] = ccl1["apply::out_labels"];
        features1["filterf::in_n_RoIs"] = ccl1["apply::out_n_RoIs"];
        features1["filterf::in_RoIs"] = cca1["extract::out_RoIs"];
        delayer["memorize::in"] = video["generate::out_img_gray8"];
        // ----------------------------- //
        // -- Associations (t - 1, t) -- //
        // ----------------------------- //

        // step 6: k-NN matching (RoIs associations)
        knn["match::in_RoIs0"] = features0["filterf::out_RoIs"];
        knn["match::in_n_RoIs0"] = features0["filterf::out_n_RoIs"];
        knn["match::in_RoIs1"] = features1["filterf::out_RoIs"];
        knn["match::in_n_RoIs1"] = features1["filterf::out_n_RoIs"];

        // step 7: temporal tracking
        tracking["perform::in_RoIs"] = knn["match::out_RoIs1"];
        tracking["perform::in_n_RoIs"] = features1["filterf::out_n_RoIs"];
        tracking["perform::in_frame"] = video["generate::out_frame"];

        // ---------- //
        // -- LOGS -- //
        // ---------- //

        // TIME_POINT(log_b);
        // save frames (CCs)
        if (p_ccl_fra_path)
        {
                (*log_fra)["write::in_labels"] = features1["filterf::fwd_labels"];
                (*log_fra)["write::in_RoIs"] = knn["match::out_RoIs1"];
                (*log_fra)["write::in_n_RoIs"] = features1["filterf::out_n_RoIs"];
        }

        // save stats
        if (p_log_path)
        {
                log_RoIs["write::in_RoIs0"] = knn["match::out_RoIs0"];
                log_RoIs["write::in_n_RoIs0"] = features0["filterf::out_n_RoIs"];
                log_RoIs["write::in_RoIs1"] = knn["match::out_RoIs1"];
                log_RoIs["write::in_n_RoIs1"] = features1["filterf::out_n_RoIs"];
                log_RoIs["write::in_frame"] = video["generate::out_frame"];

                // if (cur_fra > (uint32_t)p_vid_in_start) {
                log_kNN["write::in_nearest"].bind(knn_data->nearest[0]);
                log_kNN["write::in_distances"].bind(knn_data->distances[0]);
#ifdef MOTION_ENABLE_DEBUG
                log_kNN["write::in_conflicts"].bind(knn_data->conflicts);
#endif
                log_kNN["write::in_RoIs0"] = knn["match::out_RoIs0"];
                log_kNN["write::in_n_RoIs0"] = features0["filterf::out_n_RoIs"];
                log_kNN["write::in_RoIs1"] = knn["match::out_RoIs1"];
                log_kNN["write::in_n_RoIs1"] = features1["filterf::out_n_RoIs"];
                log_kNN["write::in_frame"] = video["generate::out_frame"];

                log_trk["write::in_frame"] = video["generate::out_frame"];
                //}
        }

        if (visu)
        {
                (*visu)["display::in_frame"] = video["generate::out_frame"];
                (*visu)["display::in_img"] = video["generate::out_img_gray8"];
                (*visu)["display::in_RoIs"] = knn["match::out_RoIs1"];
                (*visu)["display::in_n_RoIs"] = features1["filterf::out_n_RoIs"];
                (*visu)("display").exec();
        }

        // -------------- //
        // -- Pipeline -- //
        // -------------- //

        std::vector<stage> pip_stages = {
            make_stage(
                {
                    &delayer("produce"),
                    &video("generate"),
                },
                {
                        &sd0("compute"), 
                        &sd1("compute"),
                },
                {}),
            make_stage(
                {
                    &morpho0("computef"),
                    &morpho1("computef"),
                },
                {&knn("match")},
                {}),
            make_stage({&tracking("perform")}, {}, {})};


        if (p_ccl_fra_path)
        {
                std::get<2>(pip_stages[0]).push_back(&(*log_fra)("write"));
                std::get<2>(pip_stages[1]).push_back(&(*log_fra)("write"));
                std::get<0>(pip_stages[2]).push_back(&(*log_fra)("write"));
        }
        if (p_log_path)
        {
                std::get<2>(pip_stages[0]).push_back(&log_RoIs("write"));
                std::get<2>(pip_stages[1]).push_back(&log_RoIs("write"));
                std::get<0>(pip_stages[2]).push_back(&log_RoIs("write"));

                std::get<2>(pip_stages[0]).push_back(&log_kNN("write"));
                std::get<2>(pip_stages[1]).push_back(&log_kNN("write"));
                std::get<0>(pip_stages[2]).push_back(&log_kNN("write"));

                std::get<2>(pip_stages[0]).push_back(&log_trk("write"));
                std::get<2>(pip_stages[1]).push_back(&log_trk("write"));
                std::get<0>(pip_stages[2]).push_back(&log_trk("write"));

        }

        if (visu)
        {
                std::get<2>(pip_stages[0]).push_back(&(*visu)("display"));
                std::get<2>(pip_stages[1]).push_back(&(*visu)("display"));
                std::get<0>(pip_stages[2]).push_back(&(*visu)("display"));
        }

        std::vector<spu::runtime::Task *> pipe_first_tasks = {&video("generate"), &delayer("produce")};
        spu::runtime::Pipeline pip(pipe_first_tasks, pip_stages,
                                   {1, 2, 1},             // number of threads per stage -> one thread per stage
                                   {1, 1},              // buffer size between stages -> size 1 between stage 1 and 2
                                   {true, true},        // active waiting between stage 1 and stage 2 -> no
                                   {false, false, false}, // enable pinnig -> no
                                   {"PU0|PU1|PU2"});      // pinning to threads -> ignored because pinning is disabled
        if(p_stats)
        for(auto& seq: pip.get_stages())    
        for(auto& mdl : seq->get_modules<spu::module::Module>(false))
        for(auto& tsk : mdl->tasks)
                tsk->set_stats(true);

        TIME_POINT(start_compute);
        pip.exec([&n_processed_frames, &n_moving_objs, tracking_data, &video]() {
                n_processed_frames++;
                n_moving_objs = tracking_count_objects(tracking_data->tracks);
                fprintf(stderr, "(II) Frame n°%4d", (unsigned long)n_processed_frames);
                fprintf(stderr, " -- Tracks = %3lu\r", (unsigned long)n_moving_objs);
                fflush(stderr);
                return video.is_done();
        });
        TIME_POINT(stop_compute);


        const bool ordered = true, display_throughput = false;
        if(p_stats){
                auto stages = pip.get_stages();
                for (size_t s = 0; s < stages.size(); s++) {
                        const int n_threads = stages[s]->get_n_threads();
                        std::cout << "#" << std::endl << "# Pipeline stage " << (s + 1) << " ("
                                << n_threads << " thread(s)): " << std::endl;

                        spu::tools::Stats::show(stages[s]->get_tasks_per_types(), ordered, display_throughput);
                }
        }
                    
        std::ofstream file("graph.dot");
        pip.export_dot(file);

        n_moving_objs = tracking_count_objects(tracking_data->tracks);

        if (p_trk_roi_path)
        {
                FILE *f = fopen(p_trk_roi_path, "w");
                if (f == NULL)
                {
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

        // some frames have been buffered for the visualization, display or write these frames here
        if (visu)
                visu->flush();

        // ---------- //
        // -- FREE -- //
        // ---------- //
        
        kNN_free_data(knn_data);
        tracking_free_data(tracking_data);

        printf("#\n");
        printf("# End of the program, exiting.\n");

        return EXIT_SUCCESS;
}
