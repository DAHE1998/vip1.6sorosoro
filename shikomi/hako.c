/* hako.c — 统一容器解析（基于 FFmpeg 9.0 libavformat，静态链接，自包含）
 * 一个文件解析 FFmpeg 支持的所有容器：mp4/mov/mkv/webm/avi/flv/mpegts/...。
 * 只管视觉轨；音频不进 C（领导 2026-08-06 定）。
 *
 * 编译时通过 -I shikomi/ffmpeg_static/include 找到 libavformat/avformat.h 等，
 * 链接 -L shikomi/ffmpeg_static/lib -lavformat -lavcodec -lavutil -lz -lm -lpthread -ldl。
 *
 * 参考：FFmpeg 9.0 libavformat/{utils.c,movec.c,matroskadec.c} 的 hako 范式，
 *       annexb 部分参考 libavcodec/bsf/h264_mp4toannexb.c。 */
#include "hako.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/log.h>

/* H.264 NAL 类型（annexb 用） */
#define H264_NAL_SPS          7
#define H264_NAL_PPS          8
#define H264_NAL_SEI          6
#define H264_NAL_IDR_SLICE    5
#define SEI_TYPE_BUFFERING_PERIOD 0x03

/* ---------------- 容器解析（libavformat） ---------------- */

int mov_open(MOV *m, const char *path) {
    memset(m, 0, sizeof(*m));
    AVFormatContext *fmt = NULL;

    /* 关掉 ffmpeg 的日志噪声，保持我们自己的输出干净 */
    av_log_set_level(AV_LOG_ERROR);

    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) {
        fprintf(stderr, "hako: avformat_open_input 失败: %s\n", path);
        return -1;
    }
    if (avformat_find_stream_info(fmt, NULL) < 0) {
        fprintf(stderr, "hako: avformat_find_stream_info 失败: %s\n", path);
        avformat_close_input(&fmt);
        return -1;
    }

    int vidx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        AVStream *st = fmt->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            /* 优先选 H.264（我们解码器固定 cudaVideoCodec_H264） */
            if (st->codecpar->codec_id == AV_CODEC_ID_H264) { vidx = (int)i; break; }
            if (vidx < 0) vidx = (int)i;
        }
    }
    if (vidx < 0) {
        fprintf(stderr, "hako: 未找到视频轨: %s\n", path);
        avformat_close_input(&fmt);
        return -1;
    }

    AVStream *vst = fmt->streams[vidx];
    m->fmtctx = fmt;
    m->vst = vst;
    m->vstream_idx = vidx;
    m->has_video = 1;
    m->width  = vst->codecpar->width;
    m->height = vst->codecpar->height;

    /* 帧率：优先 avg_frame_rate，回退 r_frame_rate（MKV/ts 常靠这个），
       再不行用 time_base 估算。 */
    AVRational fr = vst->avg_frame_rate;
    if (fr.num <= 0 || fr.den <= 0) fr = vst->r_frame_rate;
    if (fr.num <= 0 || fr.den <= 0) fr = av_inv_q(vst->time_base);
    if (fr.num > 0 && fr.den > 0) {
        m->v_fps_num = fr.num;
        m->v_fps_den = fr.den;
    } else {
        m->v_fps_num = 30000; m->v_fps_den = 1001;   /* 最后兜底 29.97 */
    }

    /* 时长（微秒域）：用容器级 fmt->duration（AV_TIME_BASE=1e6） */
    m->m_timescale = AV_TIME_BASE;
    m->m_duration  = fmt->duration;
    m->v_timescale = vst->time_base.den;
    m->v_duration  = vst->duration;
    m->v_nb_frames = vst->nb_frames;
    m->v_n = (int)vst->nb_frames;
    /* MKV/ts 常无 nb_frames：用 duration(流时基秒) × fps 估算 */
    if (m->v_nb_frames <= 0 && vst->duration > 0) {
        double dur_sec = (double)vst->duration * av_q2d(vst->time_base);
        m->v_nb_frames = (int)llround(dur_sec * (double)m->v_fps_num / m->v_fps_den);
        m->v_n = m->v_nb_frames;
    }

    /* extradata：H.264 为 avcC（MP4），MKV 同样为 avcC；喂给 annexb_open */
    if (vst->codecpar->extradata && vst->codecpar->extradata_size > 0) {
        m->v_extradata_size = vst->codecpar->extradata_size;
        m->v_extradata = (uint8_t *)malloc((size_t)m->v_extradata_size);
        if (m->v_extradata)
            memcpy(m->v_extradata, vst->codecpar->extradata, (size_t)m->v_extradata_size);
    }

    /* 分配游标 packet */
    m->pkt = av_packet_alloc();
    if (!m->pkt) {
        fprintf(stderr, "hako: av_packet_alloc 失败\n");
        mov_close(m);
        return -1;
    }
    return 0;
}

void mov_close(MOV *m) {
    if (!m) return;
    if (m->pkt) { av_packet_free((AVPacket **)&m->pkt); m->pkt = NULL; }
    if (m->fmtctx) { avformat_close_input((AVFormatContext **)&m->fmtctx); m->fmtctx = NULL; }
    m->vst = NULL;
    free(m->v_extradata); m->v_extradata = NULL;
    free(m->v); m->v = NULL;
    free((void *)m->map); m->map = NULL;
    if (m->fd > 0) { /* 旧 mmap 路径已不使用，保留防御 */ }
    m->fd = 0;
}

int mov_read_next(MOV *m, uint8_t *buf, long *out_n) {
    if (!m || !m->fmtctx || !m->pkt) return -1;
    AVFormatContext *fmt = (AVFormatContext *)m->fmtctx;
    AVPacket *pkt = (AVPacket *)m->pkt;

    for (;;) {
        int r = av_read_frame(fmt, pkt);
        if (r < 0) {
            /* r==AVERROR_EOF 正常结束；其他为错误 */
            if (r == AVERROR_EOF) return 0;
            return -1;
        }
        if (pkt->stream_index != m->vstream_idx) {
            av_packet_unref(pkt);
            continue;  /* 跳过音频/字幕等非视频包 */
        }
        /* 复制视频帧数据到调用方缓冲 */
        if (pkt->size > 0 && buf) memcpy(buf, pkt->data, (size_t)pkt->size);
        *out_n = (long)pkt->size;
        /* 注意：不在此 unref，调用方读完后再 mov_read_next 会覆盖 pkt；
           为保证安全，这里读取后立即 unref，数据已复制。 */
        av_packet_unref(pkt);
        return 1;
    }
}

/* 兼容旧签名：忽略 s，顺序取下一帧 */
long mov_read_sample(const MOV *m, const MOVSample *s, uint8_t *buf) {
    (void)s;
    long n = 0;
    int r = mov_read_next((MOV *)m, buf, &n);
    if (r <= 0) return -1;
    return n;
}

/* ---------------- H.264 avcC → annexb（抄 FFmpeg 9.0 h264_mp4toannexb） ---------------- */

int annexb_open(AnnexB *s, const uint8_t *extradata, int extradata_size) {
    uint16_t unit_size;
    uint32_t total_size = 0;
    uint8_t *out = NULL, sps_done = 0;
    static const uint8_t nalu_header[4] = { 0, 0, 0, 1 };
    int length_size, pps_offset = 0;
    const uint8_t *p = extradata, *e = extradata + extradata_size;

    if (extradata_size < 7) return -1;
    p += 4;                                          /* configurationVersion + profile + compat + level */
    length_size = (*p & 0x3) + 1;                    /* lengthSizeMinusOne */
    p++;
    {
        int unit_nb = *p & 0x1f;                     /* numOfSequenceParameterSets */
        p++;
        while (unit_nb-- && p + 2 <= e) {
            unit_size = (uint16_t)((p[0] << 8) | p[1]);
            p += 2;
            if (unit_size > e - p) { free(out); return -1; }
            out = realloc(out, total_size + unit_size + 4);
            if (!out) return -1;
            memcpy(out + total_size, nalu_header, 4);
            memcpy(out + total_size + 4, p, unit_size);
            total_size += unit_size + 4;
            p += unit_size;
            if (!unit_nb && !sps_done++) {           /* PPS 计数 */
                unit_nb = *p & 0x1f;
                p++;
                pps_offset = (int)total_size;
            }
        }
    }
    s->sps_size = pps_offset;
    s->sps = malloc((size_t)pps_offset ? (size_t)pps_offset : 1);
    if (pps_offset && s->sps) memcpy(s->sps, out, pps_offset);
    s->pps_size = total_size - pps_offset;
    s->pps = malloc((size_t)(total_size - pps_offset) ? (size_t)(total_size - pps_offset) : 1);
    if (s->pps && s->pps_size) memcpy(s->pps, out + pps_offset, s->pps_size);
    free(out);
    s->length_size = length_size;
    s->new_idr = 1;
    s->idr_sps_seen = 0;
    s->idr_pps_seen = 0;
    s->extradata_parsed = 1;
    return 0;
}

/* 抄 count_or_copy：startcode 判定（PS_OUT_OF_BAND=0 / 首个=4 / 其余=3） */
static void count_or_copy(uint8_t **out, uint64_t *out_size,
                          const uint8_t *in, int in_size, int ps, int copy) {
    int start_code_size;
    if (ps == -1)              start_code_size = 0;
    else if (ps == 1 || *out_size == 0) start_code_size = 4;
    else                       start_code_size = 3;
    if (copy) {
        memcpy(*out + start_code_size, in, in_size);
        if (start_code_size == 4) {
            (*out)[0] = (*out)[1] = (*out)[2] = 0; (*out)[3] = 1;
        } else if (start_code_size) {
            (*out)[0] = (*out)[1] = 0; (*out)[2] = 1;
        }
        *out += start_code_size + in_size;
    }
    *out_size += (uint64_t)start_code_size + in_size;
}

void annexb_close(AnnexB *s) {
    free(s->sps); free(s->pps);
    s->sps = s->pps = NULL;
    s->sps_size = s->pps_size = 0;
}

/* 抄 h264_mp4toannexb_filter（单 pass 动态缓冲，输出逐位一致） */
int annexb_filter(AnnexB *s, const uint8_t *buf_in, int buf_size,
                         uint8_t **out_buf, int *out_size) {
    const uint8_t *buf_end = buf_in + buf_size;
    const uint8_t *buf;
    uint8_t *op;
    uint64_t out_size2 = 0;
    uint8_t unit_type, new_idr, sps_seen, pps_seen;

    if (!s->extradata_parsed) { *out_buf = NULL; *out_size = 0; return 0; }

    buf = buf_in;
    new_idr = s->new_idr; sps_seen = s->idr_sps_seen; pps_seen = s->idr_pps_seen;
    while (buf < buf_end) {
        uint32_t nal_size = 0;
        for (int i = 0; i < s->length_size; i++)
            nal_size = (nal_size << 8) | buf[i];
        buf += s->length_size;
        if ((int64_t)nal_size > buf_end - buf) return -1;
        if (!nal_size) continue;
        unit_type = *buf & 0x1f;
        if (unit_type == H264_NAL_SPS)      sps_seen = new_idr = 1;
        else if (unit_type == H264_NAL_PPS) {
            pps_seen = new_idr = 1;
            if (!sps_seen && s->sps_size)
                count_or_copy(NULL, &out_size2, s->sps, s->sps_size, -1, 0), sps_seen = 1;
        }
        if (!new_idr && unit_type == H264_NAL_IDR_SLICE && (buf[1] & 0x80))
            new_idr = 1;
        if (unit_type == H264_NAL_SEI && buf[1] == SEI_TYPE_BUFFERING_PERIOD && !sps_seen && !pps_seen) {
            if (s->sps_size) count_or_copy(NULL, &out_size2, s->sps, s->sps_size, -1, 0), sps_seen = 1;
            if (s->pps_size) count_or_copy(NULL, &out_size2, s->pps, s->pps_size, -1, 0), pps_seen = 1;
        }
        if (new_idr && unit_type == H264_NAL_IDR_SLICE && !sps_seen && !pps_seen) {
            if (s->sps_size) count_or_copy(NULL, &out_size2, s->sps, s->sps_size, -1, 0);
            if (s->pps_size) count_or_copy(NULL, &out_size2, s->pps, s->pps_size, -1, 0);
            new_idr = 0;
        } else if (new_idr && unit_type == H264_NAL_IDR_SLICE && sps_seen && !pps_seen) {
            if (s->pps_size) count_or_copy(NULL, &out_size2, s->pps, s->pps_size, -1, 0);
        }
        count_or_copy(NULL, &out_size2, buf, (int)nal_size, unit_type, 0);
        buf += nal_size;
    }

    *out_buf = (uint8_t *)malloc((size_t)out_size2 + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!*out_buf) return -1;
    op = *out_buf;
    out_size2 = 0;
    buf = buf_in;
    new_idr = s->new_idr; sps_seen = s->idr_sps_seen; pps_seen = s->idr_pps_seen;
    while (buf < buf_end) {
        uint32_t nal_size = 0;
        for (int i = 0; i < s->length_size; i++)
            nal_size = (nal_size << 8) | buf[i];
        buf += s->length_size;
        if ((int64_t)nal_size > buf_end - buf) { free(*out_buf); *out_buf = NULL; return -1; }
        if (!nal_size) continue;
        unit_type = *buf & 0x1f;
        if (unit_type == H264_NAL_SPS)      sps_seen = new_idr = 1;
        else if (unit_type == H264_NAL_PPS) {
            pps_seen = new_idr = 1;
            if (!sps_seen && s->sps_size)
                count_or_copy(&op, &out_size2, s->sps, s->sps_size, -1, 1), sps_seen = 1;
        }
        if (!new_idr && unit_type == H264_NAL_IDR_SLICE && (buf[1] & 0x80))
            new_idr = 1;
        if (unit_type == H264_NAL_SEI && buf[1] == SEI_TYPE_BUFFERING_PERIOD && !sps_seen && !pps_seen) {
            if (s->sps_size) count_or_copy(&op, &out_size2, s->sps, s->sps_size, -1, 1), sps_seen = 1;
            if (s->pps_size) count_or_copy(&op, &out_size2, s->pps, s->pps_size, -1, 1), pps_seen = 1;
        }
        if (new_idr && unit_type == H264_NAL_IDR_SLICE && !sps_seen && !pps_seen) {
            if (s->sps_size) count_or_copy(&op, &out_size2, s->sps, s->sps_size, -1, 1);
            if (s->pps_size) count_or_copy(&op, &out_size2, s->pps, s->pps_size, -1, 1);
            new_idr = 0;
        } else if (new_idr && unit_type == H264_NAL_IDR_SLICE && sps_seen && !pps_seen) {
            if (s->pps_size) count_or_copy(&op, &out_size2, s->pps, s->pps_size, -1, 1);
        }
        count_or_copy(&op, &out_size2, buf, (int)nal_size, unit_type, 1);
        buf += nal_size;
    }
    memset(*out_buf + out_size2, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    *out_size = (int)out_size2;
    s->new_idr = new_idr;
    s->idr_sps_seen = sps_seen;
    s->idr_pps_seen = pps_seen;
    return 0;
}
