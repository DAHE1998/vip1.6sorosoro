/* hako.h — 统一容器解析（基于 FFmpeg 9.0 libavformat，静态链接，自包含）
 * 一个 C 文件解析 FFmpeg 支持的所有容器（mp4/mov/mkv/webm/avi/flv/mpegts...）。
 * 只管视觉：视频轨 + 帧数据 + avcC→annexb；音频不进 C（领导 2026-08-06 定）。
 * 静态库与头文件随项目发布于 shikomi/ffmpeg_static/，不依赖系统 ffmpeg。 */
#ifndef DEMUX_H
#define DEMUX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t off;
    uint32_t size;
} MOVSample;

/* 视频轨编码类型（容器无关；NVDEC 可硬解集合见 kaiseki 映射） */
typedef enum {
    HAKO_CODEC_UNKNOWN = 0,
    HAKO_CODEC_H264,
    HAKO_CODEC_HEVC,
    HAKO_CODEC_AV1,
    HAKO_CODEC_VP9,
    HAKO_CODEC_VP8,
    HAKO_CODEC_MPEG2,
    HAKO_CODEC_VC1,
} HakoCodec;

typedef struct {
    /* 文件/mmap 信息（libavformat 模式下 off/size/v 不再作为数据来源，仅兼容旧字段） */
    int          fd;
    uint64_t     fsize;
    const uint8_t *map;

    uint32_t     m_timescale;
    int64_t      m_duration;

    int          has_video;
    HakoCodec    v_codec;      /* 视频轨编码（2026-09-01：多编码 NVDEC，不再 H264 专用） */
    uint32_t     width, height;
    uint32_t     v_timescale;
    int64_t      v_nb_frames;
    int64_t      v_duration;
    int          v_extradata_size;
    uint8_t     *v_extradata;
    int          v_n;          /* 估计总帧数（用于日志/进度，非严格随机访问索引） */
    MOVSample   *v;            /* 旧接口兼容占位，libavformat 模式为 NULL */
    int          v_fps_num, v_fps_den;

    /* libavformat 上下文（不透明，避免在此暴露 ffmpeg 类型） */
    void        *fmtctx;       /* AVFormatContext* */
    void        *vst;          /* AVStream* */
    int          vstream_idx;
    void        *pkt;          /* AVPacket* 游标 */
} MOV;

/* 打开容器，定位视频轨。path 为本地文件。成功返回 0，失败 <0 */
int  mov_open(MOV *m, const char *path);
void mov_close(MOV *m);

/* 顺序读取下一视频帧：成功返回 1 且 *out_n 为字节数；结束返回 0；错误 <0。
 * buf 需调用方预分配 >= 单帧最大尺寸（建议 64MB，与 kaiseki 的 g_sam 一致）。 */
int  mov_read_next(MOV *m, uint8_t *buf, long *out_n);

/* 兼容旧接口：忽略 s，内部顺序取下一帧。仅保留以便过渡，新代码请用 mov_read_next。 */
long mov_read_sample(const MOV *m, const MOVSample *s, uint8_t *buf);

/* H.264 avcC → annexb（参考自 FFmpeg 9.0 libavcodec/bsf/h264_mp4toannexb.c） */
typedef struct {
    uint8_t *sps; int sps_size;
    uint8_t *pps; int pps_size;
    int length_size;
    int new_idr, idr_sps_seen, idr_pps_seen, extradata_parsed;
} AnnexB;

int  annexb_open(AnnexB *ab, const uint8_t *avcc, int avcc_size);
void annexb_close(AnnexB *ab);
int  annexb_filter(AnnexB *ab, const uint8_t *in, int in_size, uint8_t **out, int *out_size);

/* H.265 hvcC → annexb（参考自 FFmpeg 9.0 libavcodec/bsf/hevc_mp4toannexb.c）。
 * 与 H264 版同构：length 前缀 NAL → startcode，首帧前补 hvcC 里的 VPS/SPS/PPS。 */
typedef struct {
    uint8_t *vps; int vps_size;
    uint8_t *sps; int sps_size;
    uint8_t *pps; int pps_size;
    int length_size;
    int vps_seen, sps_seen, pps_seen, extradata_parsed;
} HevcAnnexB;

int  hevc_annexb_open(HevcAnnexB *hb, const uint8_t *hvcC, int hvcC_size);
void hevc_annexb_close(HevcAnnexB *hb);
int  hevc_annexb_filter(HevcAnnexB *hb, const uint8_t *in, int in_size, uint8_t **out, int *out_size);

#ifdef __cplusplus
}
#endif

#endif
