/* mp4_mov.h — 自写 MP4/MOV 容器解析（参考自 FFmpeg 9.0 源码 libavformat/mov.c 等，见 mp4_mov.c 头注释）
 * 只管视觉：视频轨样本表 + 元数据 + avcC→annexb；音频不进 C（领导 2026-08-06 定） */
#ifndef MP4_MOV_H
#define MP4_MOV_H

#include <stdint.h>

typedef struct {
    uint64_t off;
    uint32_t size;
} MOVSample;

typedef struct {
    int          fd;
    uint64_t     fsize;
    const uint8_t *map;

    uint32_t     m_timescale;
    int64_t      m_duration;

    int          has_video;
    uint32_t     width, height;
    uint32_t     v_timescale;
    int64_t      v_nb_frames;
    int64_t      v_duration;
    int          v_extradata_size;
    uint8_t     *v_extradata;
    int          v_n;
    MOVSample   *v;
    int          v_fps_num, v_fps_den;
} MOV;

int  mov_open(MOV *m, const char *path);
void mov_close(MOV *m);
long mov_read_sample(const MOV *m, const MOVSample *s, uint8_t *buf);

/* H.264 avcC → annexb（参考自 FFmpeg 9.0 源码 libavcodec/bsf/h264_mp4toannexb.c） */
typedef struct {
    uint8_t *sps; int sps_size;
    uint8_t *pps; int pps_size;
    int length_size;
    int new_idr, idr_sps_seen, idr_pps_seen, extradata_parsed;
} AnnexB;

int  annexb_open(AnnexB *ab, const uint8_t *avcc, int avcc_size);
void annexb_close(AnnexB *ab);
int  annexb_filter(AnnexB *ab, const uint8_t *in, int in_size, uint8_t **out, int *out_size);

#endif
