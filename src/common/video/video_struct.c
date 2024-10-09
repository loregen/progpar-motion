#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "motion/video/video_struct.h"

enum video_codec_e video_str_to_enum(const char* str) {
    if (strcmp(str, "FFMPEG-IO") == 0) {
        return VCDC_FFMPEG_IO;
    } else {
        fprintf(stderr, "(EE) '%s()' failed, unknow input ('%s').\n", __func__, str);
        exit(-1);
    }
}

enum video_codec_hwaccel_e video_hwaccel_str_to_enum(const char* str) {
    if (strcmp(str, "NONE") == 0) {
        return VCDC_HWACCEL_NONE;
    } else if (strcmp(str, "NVDEC") == 0) {
        return VCDC_HWACCEL_NVDEC;
    } else if (strcmp(str, "VIDTB") == 0) {
        return VCDC_HWACCEL_VIDEOTOOLBOX;
    } else {
        fprintf(stderr, "(EE) '%s()' failed, unknow input ('%s').\n", __func__, str);
        exit(-1);
    }
}
