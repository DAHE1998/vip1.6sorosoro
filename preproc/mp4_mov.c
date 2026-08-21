/*
 * preproc/mp4_mov.c — 自写 MP4/MOV 容器解析 + H.264 avcC→annexb 重写（只管视觉，无音频）
 *
 * 用法: 随 duo_analyze 链接（mp4_mov.h 暴露 mov_open/mov_read_sample/annexb_*），不单独执行
 * 依赖: 输入 mp4/mov 文件（mmap 全量遍历 box）；参考自 FFmpeg 9.0 源码（mov.c / h264_mp4toannexb.c /
 *       rational.c，自写 C 内化，不链接 libavformat/libavcodec）
 * 产物: 视频轨元数据（宽高/时间基准/总帧数/时长/avcC extradata）+ 样本表 (offset,size)
 *       （stsc×stsz×stco 展开，压缩顺序 = demux 顺序）+ H.264 annexb 流（IDR 注入 SPS/PPS）
 *
 * 音频：领导 2026-08-06 定「一条 ffmpeg 命令提取，不进 C」，本文件不解析音频轨。
 * 踩坑：box size 用大端 rb32、4CC 用小端 rl32（照抄 mov.c），混用会全部比较失败。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define MKTAG(a,b,c,d) ((a) | ((b)<<8) | ((c)<<16) | ((d)<<24))
#define MAX_STSC_ENTRIES 65536
#define MAX_ATOM_DEPTH 10
#define MAX_STTS_DELTA (UINT32_MAX - 48000 * 10)  /* mov.c max_stts_delta 默认值 */

typedef struct { uint32_t first, count, id; } StscEntry;
typedef struct { uint32_t count, duration; } SttsEntry;

typedef struct {
    uint64_t off;
    uint32_t size;
} MOVSample;

typedef struct {
    /* 文件（mmap） */
    int          fd;
    uint64_t     fsize;
    const uint8_t *map;

    /* movie（mvhd） */
    uint32_t     m_timescale;
    int64_t      m_duration;

    /* 视频轨 */
    int          has_video;
    uint32_t     width, height;         /* stsd video entry（显示尺寸） */
    uint32_t     v_timescale;           /* mdhd */
    int64_t      v_nb_frames;           /* stts 总样本数 */
    int64_t      v_duration;            /* stts duration 累加（fps 分母） */
    int          v_extradata_size;
    uint8_t     *v_extradata;           /* avcC 原样 */
    int          v_n;                   /* 样本数 */
    MOVSample   *v;                     /* [v_n] 压缩顺序 */
    int          v_fps_num, v_fps_den;  /* av_reduce 后 fps（=ffmpeg avg_frame_rate） */
} MOV;

/* ── 大端读（对应 avio_r8/rb16/rb32/rb64） ────────────── */
static inline uint8_t  rd8 (const uint8_t *p) { return p[0]; }
static inline uint16_t rd16(const uint8_t *p) { return (uint16_t)((p[0]<<8)|p[1]); }
static inline uint32_t rd32(const uint8_t *p) { return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3]; }
/* 4CC 用 avio_rl32（小端读）读，配 MKTAG 小端拼——照抄 mov.c：size 是 rb32、
 * type 是 rl32（`type = avio_rl32(pb)`），混用会全部比较失败（已踩坑） */
static inline uint32_t rl32(const uint8_t *p) { return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }
static inline uint64_t rd64(const uint8_t *p) {
    return ((uint64_t)rd32(p)<<32) | rd32(p+4);
}

/* ── 抄 libavutil/rational.c av_reduce（fps 约分） ─────── */
static int64_t av_gcd_(int64_t a, int64_t b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b) { int64_t t = a % b; a = b; b = t; }
    return a;
}
static int av_reduce_(int *dst_num, int *dst_den, int64_t num, int64_t den, int64_t max) {
    int a0n = 0, a0d = 1, a1n = 1, a1d = 0;
    int sign = (num < 0) ^ (den < 0);
    int64_t gcd = av_gcd_(num, den);
    if (gcd) { num = (num < 0 ? -num : num) / gcd; den = (den < 0 ? -den : den) / gcd; }
    if (num <= max && den <= max) { a1n = (int)num; a1d = (int)den; den = 0; }
    while (den) {
        int64_t x = num / den;
        int64_t next_den = num - den * x;
        int64_t a2n = x * a1n + a0n;
        int64_t a2d = x * a1d + a0d;
        if (a2n > max || a2d > max) {
            if (a1n) x = (max - a0n) / a1n;
            if (a1d) { int64_t x2 = (max - a0d) / a1d; if (x2 < x) x = x2; }
            if (den * (2 * x * a1d + a0d) > num * a1d) { a1n = (int)(x * a1n + a0n); a1d = (int)(x * a1d + a0d); }
            break;
        }
        a0n = a1n; a0d = a1d;
        a1n = (int)a2n; a1d = (int)a2d;
        num = den; den = next_den;
    }
    *dst_num = sign ? -a1n : a1n;
    *dst_den = a1d;
    return den == 0;
}

/* ═══════════════════════════════════════════════════════
 * box 解析（抄 mov_read_default：size==1 扩展 64bit、size==0 到父末尾、
 * 深度上限、非法子 box 跳过）
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    const uint8_t *b, *e;   /* 当前 box 数据区 [b,e) */
    uint32_t type;          /* 4CC（大端序如 MKTAG 值） */
} Atom;

/* 遍历父 box 内容：对每个子 box 调 cb(type, data, size, user) */
typedef void (*atom_cb)(uint32_t type, const uint8_t *d, uint64_t sz, void *user);

static void for_each_atom(const uint8_t *d, uint64_t sz, int depth, atom_cb cb, void *user) {
    uint64_t total = 0;
    if (depth > MAX_ATOM_DEPTH) return;
    while (total + 8 <= sz) {
        const uint8_t *p = d + total;
        uint64_t box_size = rd32(p);
        uint32_t type = rl32(p + 4);
        uint64_t hdr = 8;
        if (box_size == 1) {                 /* 64 位扩展大小 */
            if (total + 16 > sz) break;
            box_size = rd64(p + 8);
            hdr = 16;
        }
        if (box_size == 0) box_size = sz - total;   /* size 0 = 到父末尾 */
        if (box_size < hdr || box_size - hdr > sz - total) break;  /* 非法/越界 */
        cb(type, p + hdr, box_size - hdr, user);
        total += box_size;
    }
}

/* ═══════════════════════════════════════════════════════
 * trak 解析（抄 mov_read_trak/tkhd/mdhd/hdlr + stbl 各表）
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    MOV *m;
    int  is_video;
    int  in_stbl;
    uint32_t v_timescale; int64_t v_duration;
    int64_t nb_frames_for_fps, duration_for_fps;
    int64_t stts_duration;
    /* stbl 表 */
    StscEntry *stsc; int stsc_count;
    uint32_t stsz_sample_size; uint32_t *sample_sizes; int sample_count; uint64_t data_size;
    uint64_t *chunk_offsets; int chunk_count;
    SttsEntry *stts; int stts_count;
} Track;

/* ── 抄 mov_read_stsc ─────────────────────────────────── */
static void parse_stsc(Track *t, const uint8_t *d, uint64_t sz) {
    unsigned int i, entries;
    if (sz < 8) return;
    entries = rd32(d + 4);   /* 前 4B = version/flags */
    if ((uint64_t)entries * 12 + 4 > sz) return;
    if (!entries) return;
    if (entries > MAX_STSC_ENTRIES) return;
    t->stsc = malloc(entries * sizeof(StscEntry));
    if (!t->stsc) return;
    t->stsc_count = 0;
    for (i = 0; i < entries && 8 + (uint64_t)i * 12 + 12 <= sz; i++) {
        const uint8_t *p = d + 8 + i * 12;
        t->stsc[i].first = rd32(p);
        t->stsc[i].count = rd32(p + 4);
        t->stsc[i].id    = rd32(p + 8);
    }
    t->stsc_count = i;
    /* 抄 mov.c 3426-3452：非法条目修复循环 */
    for (i = t->stsc_count - 1; i < UINT_MAX; i--) {
        int64_t first_min = i + 1;
        if ((i + 1 < (unsigned)t->stsc_count && t->stsc[i].first >= t->stsc[i + 1].first) ||
            (i > 0 && t->stsc[i].first <= t->stsc[i - 1].first) ||
            t->stsc[i].first < first_min || t->stsc[i].count < 1 || t->stsc[i].id < 1) {
            if (i + 1 >= (unsigned)t->stsc_count) {
                if (t->stsc[i].count == 0 && i > 0) { t->stsc_count--; continue; }
                t->stsc[i].first = (uint32_t)(t->stsc[i].first < first_min ? first_min : t->stsc[i].first);
                if (i > 0 && t->stsc[i].first <= t->stsc[i - 1].first)
                    t->stsc[i].first = (uint32_t)(t->stsc[i - 1].first + 1);
                if (t->stsc[i].count < 1) t->stsc[i].count = 1;
                if (t->stsc[i].id < 1)    t->stsc[i].id = 1;
                continue;
            }
            /* 替换为下一个合法条目（照抄 mov.c 3446-3451） */
            t->stsc[i].first = t->stsc[i + 1].first - 1;
            t->stsc[i].count = t->stsc[i + 1].count;
            t->stsc[i].id    = t->stsc[i + 1].id;
        }
    }
}

/* ── 抄 mov_read_stsz（只支持 stsz 32bit field；stz2 跳过） ── */
static void parse_stsz(Track *t, const uint8_t *d, uint64_t sz) {
    unsigned int i, entries, sample_size;
    if (sz < 12) return;
    sample_size = rd32(d + 4);
    if (!t->stsz_sample_size) t->stsz_sample_size = sample_size;
    entries = rd32(d + 8);
    t->sample_count = (int)entries;
    if (sample_size) return;   /* 统一尺寸：样本表延迟到 build_index 生成 */
    if (!entries) return;
    t->sample_sizes = malloc((size_t)entries * sizeof(uint32_t));
    if (!t->sample_sizes) return;
    t->sample_count = 0;
    for (i = 0; i < entries && 12 + (uint64_t)i * 4 <= sz; i++) {
        t->sample_sizes[i] = rd32(d + 12 + i * 4);
        t->data_size += t->sample_sizes[i];
    }
    t->sample_count = (int)i;
}

/* ── 抄 mov_read_stco/co64 ─────────────────────────────── */
static void parse_stco(Track *t, const uint8_t *d, uint64_t sz, uint32_t type) {
    unsigned int i, entries;
    if (sz < 8) return;
    entries = rd32(d + 4);
    if (type == MKTAG('s','t','c','o')) {
        if ((uint64_t)entries * 4 > sz - 8) entries = (unsigned)((sz - 8) / 4);
    } else {
        if ((uint64_t)entries * 8 > sz - 8) entries = (unsigned)((sz - 8) / 8);
    }
    if (!entries) return;
    t->chunk_offsets = malloc((size_t)entries * sizeof(uint64_t));
    if (!t->chunk_offsets) return;
    t->chunk_count = 0;
    if (type == MKTAG('s','t','c','o'))
        for (i = 0; i < entries && 8 + (uint64_t)i * 4 + 4 <= sz; i++)
            t->chunk_offsets[i] = rd32(d + 8 + i * 4);
    else
        for (i = 0; i < entries && 8 + (uint64_t)i * 8 + 8 <= sz; i++) {
            t->chunk_offsets[i] = rd64(d + 8 + i * 8);
            if (t->chunk_offsets[i] >= 0x8000000000000000ULL) t->chunk_offsets[i] = 0; /* 负值归 0，照抄 mov.c 2759-2761 */
        }
    t->chunk_count = (int)i;
}

/* ── 抄 mov_read_stts（fps 分母用 duration/nb_frames 累加；裁剪修正照抄） ── */
static void parse_stts(Track *t, const uint8_t *d, uint64_t sz) {
    unsigned int i, entries;
    int64_t duration = 0, total_sample_count = 0, current_dts = 0, corrected_dts = 0;
    if (sz < 8) return;
    entries = rd32(d + 4);
    if (!entries) return;
    if ((uint64_t)entries > 65536 || (uint64_t)entries * 8 > sz - 8) {
        entries = (unsigned)((sz - 8) / 8);
        if (!entries) return;
    }
    t->stts = malloc((size_t)entries * sizeof(SttsEntry));
    if (!t->stts) return;
    t->stts_count = 0;
    for (i = 0; i < entries; i++) {
        unsigned int sample_count, sample_duration;
        sample_count    = rd32(d + 8 + (uint64_t)i * 8);
        sample_duration = rd32(d + 12 + (uint64_t)i * 8);
        t->stts[i].count = sample_count;
        t->stts[i].duration = sample_duration;
        /* 照抄 mov.c 3729-3750：超大 delta 裁剪 + dts 漂移修正 */
        if (sample_duration > MAX_STTS_DELTA) {
            int32_t delta_magnitude = (int32_t)sample_duration;
            t->stts[i].duration = 1;
            corrected_dts += (delta_magnitude < 0 ? (int64_t)delta_magnitude : 1) * sample_count;
        } else {
            corrected_dts += sample_duration * (uint64_t)sample_count;
        }
        current_dts += t->stts[i].duration * (uint64_t)sample_count;
        if (current_dts > corrected_dts) {
            int64_t drift = (current_dts - corrected_dts) / (sample_count ? sample_count : 1);
            uint32_t correction = (t->stts[i].duration > drift) ? (uint32_t)drift : t->stts[i].duration - 1;
            current_dts -= correction * (uint64_t)sample_count;
            t->stts[i].duration -= correction;
        }
        duration += (int64_t)t->stts[i].duration * (uint64_t)t->stts[i].count;
        total_sample_count += t->stts[i].count;
    }
    t->stts_count = (int)i;
    t->stts_duration = duration;
    if (duration > 0 && duration <= INT64_MAX - t->duration_for_fps &&
        total_sample_count <= INT_MAX - t->nb_frames_for_fps) {
        t->duration_for_fps  += duration;
        t->nb_frames_for_fps += total_sample_count;
    }
}

/* ═══════════════════════════════════════════════════════
 * stsd 解析（抄 mov_read_stsd + mov_parse_stsd_video）
 *   视频条目：16B 头（size/format/reserved6/dref2）→ 抄 mov_parse_stsd_video isom 分支
 *             → VisualSampleEntry 定长 70B → 尾部子 box（avcC 等，抄 mov_read_glbl）
 *   音频条目：不解析（领导 2026-08-06：音频不进 C）
 * ═══════════════════════════════════════════════════════ */

/* ── stsd 子 box 回调（视频尾部 avcC 等） ─────────────── */
static void stsd_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    Track *t = user;
    if (type == MKTAG('a','v','c','C') && sz >= 4) {
        if (t->m->v_extradata) return;            /* 抄 mov_read_glbl：多 glbl 忽略 */
        t->m->v_extradata = malloc((size_t)sz + 1);
        if (t->m->v_extradata) {
            memcpy(t->m->v_extradata, d, sz);
            t->m->v_extradata_size = (int)sz;
        }
    }
    /* hvcC/hev1 后续扩展 */
}

/* ── 抄 mov_read_stsd 的 entry 循环 ─────────────────────── */
static void parse_stsd(Track *t, const uint8_t *d, uint64_t sz) {
    int entries;
    if (sz < 8) return;
    entries = (int)rd32(d + 4);   /* 前 4B = version/flags */
    if (entries <= 0 || entries > (int)(sz / 8) || entries > 1024) return;
    {
        const uint8_t *p = d + 8;
        uint64_t left = sz - 8;
        for (int e = 0; e < entries && left >= 16; e++) {
            uint64_t size = rd32(p);
            /* p+4 = entry format（avc1/…），视觉-only 无需区分 entry 类型 */
            uint64_t hdr = 16;    /* size+format+reserved(4)+reserved(2)+dref(2)，照抄 ff_mov_read_stsd_entries */
            if (size == 1) { if (left < 24) break; size = rd64(p + 8); hdr = 24; }
            if (size == 0) size = left;
            if (size < hdr || size - hdr > left) break;
            const uint8_t *body = p + hdr;   /* mov_parse_stsd_video 起点 */
            uint64_t bsz = size - hdr;
            if (t->is_video) {
                /* 抄 mov_parse_stsd_video（isom 分支，mov.c 2829-2873）：
                 * isom skip 2+2+12 → width(2)/height(2) → skip 4+4+4+2 →
                 * pascal string 32B → depth(2) → 子 box（avcC） */
                if (bsz >= 40) {
                    const uint8_t *v = body;
                    v += 2 + 2 + 12;                       /* pre_defined/reserved */
                    t->m->width = rd16(v);
                    t->m->height = rd16(v + 2);
                }
                const uint8_t *sub = body + 70;            /* VisualSampleEntry 定长 70B
                                                            * (2+2+12+4+14+32+2+2；dump 实证 avcC 在 +70，
                                                            * 68 处是 pre_defined ff ff) */
                if (sub + 8 <= body + bsz)
                    for_each_atom(sub, (uint64_t)(body + bsz - sub), 3, stsd_child_cb, t);
            }   /* 音频 entry：不解析（领导：音频不进 C） */
            p += size; left -= size;
        }
    }
}

/* ── stbl 子 box 回调 ───────────────────────────────────── */
static void stbl_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    Track *t = user;
    if      (type == MKTAG('s','t','s','d')) parse_stsd(t, d, sz);
    else if (type == MKTAG('s','t','s','c')) parse_stsc(t, d, sz);
    else if (type == MKTAG('s','t','s','z')) parse_stsz(t, d, sz);
    else if (type == MKTAG('s','t','c','o') || type == MKTAG('c','o','6','4')) parse_stco(t, d, sz, type);
    else if (type == MKTAG('s','t','t','s')) parse_stts(t, d, sz);
    /* ctts/stss 不需要：Preproc 按压缩顺序喂 NVCUVID，无 seek/PTS 需求 */
}

/* ── 抄 mov_read_mdhd：timescale + duration ─────────────── */
static void parse_mdhd(Track *t, const uint8_t *d, uint64_t sz) {
    /* 照抄 mov.c mov_read_mdhd：version/flags(4) + creation(4/8) + modification(4/8)
     * 之后才是 timescale/duration（creation/modification 必须跳过） */
    if (sz >= 20) {
        int version = rd8(d);
        const uint8_t *p = d + 4;
        if (version == 1) {
            if (sz >= 32) { t->v_timescale = rd32(p + 16); t->v_duration = rd64(p + 20); }
        } else {
            if (sz >= 20) { t->v_timescale = rd32(p + 8); t->v_duration = rd32(p + 12); }
        }
        if (t->v_timescale <= 0) t->v_timescale = 1;   /* 照抄 mov.c 1952-1955 */
    }
}

/* ── 抄 mov_read_hdlr：component subtype（第 8 字节起 4B） ── */
static void parse_hdlr(Track *t, const uint8_t *d, uint64_t sz) {
    if (sz >= 12) {
        uint32_t stype = rl32(d + 8);
        if (stype == MKTAG('v','i','d','e')) t->is_video = 1;
        /* 音频 trak 不解析（领导：音频不进 C） */
    }
}

/* ── 层级递归：trak → mdia → minf → stbl（照抄 mov_read_trak/
 *    mov_read_mdia/mov_read_minf/mov_read_stbl 的嵌套） ── */
static void minf_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    if (type == MKTAG('s','t','b','l'))
        for_each_atom(d, sz, 4, stbl_child_cb, user);
}
static void mdia_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    Track *t = user;
    if      (type == MKTAG('m','d','h','d')) parse_mdhd(t, d, sz);
    else if (type == MKTAG('h','d','l','r')) parse_hdlr(t, d, sz);
    else if (type == MKTAG('m','i','n','f')) for_each_atom(d, sz, 3, minf_child_cb, t);
}
static void trak_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    if (type == MKTAG('m','d','i','a'))
        for_each_atom(d, sz, 3, mdia_child_cb, user);
}

/* ═══════════════════════════════════════════════════════
 * mov_build_index 主路径（抄 mov.c 4888-5011，简化：无 elst/ctts/
 * stss/rap/keyframe/dts——样本表只要 (offset,size) 压缩顺序）
 * ═══════════════════════════════════════════════════════ */
static void build_index(MOV *m, Track *t, MOVSample **out, int *out_n) {
    (void)m;
    uint64_t current_offset;
    unsigned int stts_index = 0, stsc_index = 0, stts_sample = 0;
    unsigned int i, j;

    /* 只支持常规路径（音频 stts duration==1 的旧 PCM 特例不支持，AAC 走常规） */
    if (!t->chunk_count || !t->sample_count || !t->stsc_count || !t->stts_count) return;

    *out = malloc((size_t)t->sample_count * sizeof(MOVSample));
    if (!*out) return;
    *out_n = 0;

    for (i = 0; i < (unsigned)t->chunk_count; i++) {
        uint64_t next_offset = i + 1 < (unsigned)t->chunk_count ? t->chunk_offsets[i + 1] : UINT64_MAX;
        current_offset = t->chunk_offsets[i];
        /* 抄 mov.c 4923-4925：stsc 推进（first 从 1 计） */
        while ((int)stsc_index + 1 < t->stsc_count && i + 1 == t->stsc[stsc_index + 1].first)
            stsc_index++;
        /* 抄 mov.c 4927-4935：stsz 尺寸校验（损坏文件时退化为 sample_sizes） */
        if (next_offset > current_offset && t->stsz_sample_size > 0 &&
            t->stsc[stsc_index].count * (int64_t)t->stsz_sample_size > (int64_t)(next_offset - current_offset)) {
            /* 照抄：stsz_sample_size 过大则忽略（日志省略） */
        }
        for (j = 0; j < t->stsc[stsc_index].count; j++) {
            unsigned int sample_size;
            if (*out_n >= t->sample_count) return;   /* 抄 mov.c 4939-4942 wrong sample count */
            sample_size = t->stsz_sample_size > 0 ? t->stsz_sample_size : t->sample_sizes[*out_n];
            (*out)[*out_n].off = current_offset;
            (*out)[*out_n].size = sample_size;
            (*out_n)++;
            current_offset += sample_size;
            /* 抄 mov.c 4999-5009：stts 按样本计数推进 */
            if (stts_index + 1 < (unsigned)t->stts_count && ++stts_sample == t->stts[stts_index].count) {
                stts_sample = 0;
                stts_index++;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * 顶层 box 遍历（抄 mov_read_header：root → moov → trak/mvhd）
 * ═══════════════════════════════════════════════════════ */
static void moov_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    MOV *m = user;
    if (type == MKTAG('m','v','h','d')) {
        /* 抄 mov_read_mvhd：movie timescale + duration（g_duration 用）；
         * version/flags(4) + creation(4/8) + modification(4/8) 之后才是 timescale/duration */
        if (sz >= 20) {
            int version = rd8(d);
            const uint8_t *p = d + 4;
            if (version == 1) {
                if (sz >= 32) { m->m_timescale = rd32(p + 16); m->m_duration = rd64(p + 20); }
            } else {
                if (sz >= 20) { m->m_timescale = rd32(p + 8); m->m_duration = rd32(p + 12); }
            }
            if (m->m_timescale <= 0) m->m_timescale = 1;   /* 照抄 mov.c 1979-1982 */
        }
    } else if (type == MKTAG('t','r','a','k')) {
        /* 抄 mov_read_trak：每个 trak 一个 Track 上下文 */
        Track *t = calloc(1, sizeof(Track));
        if (!t) return;
        t->m = m;
        for_each_atom(d, sz, 2, trak_child_cb, t);
        if (t->is_video) {
            m->has_video = 1;
            m->v_timescale = t->v_timescale;
            m->v_nb_frames = t->nb_frames_for_fps;
            m->v_duration  = t->duration_for_fps;
            build_index(m, t, &m->v, &m->v_n);
            /* fps = 抄 mov_read_header 11338-11341：av_reduce(timescale*nb, duration) */
            if (t->nb_frames_for_fps > 0 && t->duration_for_fps > 0) {
                int num, den;
                av_reduce_(&num, &den, (int64_t)t->v_timescale * t->nb_frames_for_fps,
                           t->duration_for_fps, INT_MAX);
                m->v_fps_num = num; m->v_fps_den = den;
            }
        }
        /* 音频 trak：不解析（领导：音频不进 C） */
        free(t->stsc); free(t->sample_sizes); free(t->chunk_offsets); free(t->stts);
        free(t);
    }
}

static void root_child_cb(uint32_t type, const uint8_t *d, uint64_t sz, void *user) {
    if (type == MKTAG('m','o','o','v'))
        for_each_atom(d, sz, 1, moov_child_cb, user);
    /* mdat 跳过：样本数据按 stco 偏移直接读 */
}

/* ═══════════════════════════════════════════════════════
 * mov_open：mmap + 全量解析 + demux 顺序归并
 * ═══════════════════════════════════════════════════════ */
void mov_close(MOV *m);

int mov_open(MOV *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat st;
    if (fstat(m->fd, &st) < 0 || st.st_size < 8) { close(m->fd); m->fd = -1; return -1; }
    m->fsize = (uint64_t)st.st_size;
    m->map = mmap(NULL, m->fsize, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->map == MAP_FAILED) { close(m->fd); m->fd = -1; return -1; }
    for_each_atom(m->map, m->fsize, 0, root_child_cb, m);
    if (!m->has_video) { mov_close(m); return -1; }
    /* 视频轨单独：样本表按 stco 偏移序生成（stsc/stco 保证 chunk 偏移递增），
     * v[0..v_n) 即 demux 顺序（av_read_frame 单轨等价） */
    return 0;
}

void mov_close(MOV *m) {
    if (m->map && m->map != MAP_FAILED) munmap((void *)m->map, m->fsize);
    if (m->fd >= 0) close(m->fd);
    free(m->v_extradata); free(m->v);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

/* 读样本数据到 buf（返回字节数，-1 失败） */
long mov_read_sample(const MOV *m, const MOVSample *s, uint8_t *buf) {
    if (s->off + s->size > m->fsize) return -1;
    memcpy(buf, m->map + s->off, s->size);
    return (long)s->size;
}

/* ═══════════════════════════════════════════════════════
 * H.264 avcC → annexb（抄 libavcodec/bsf/h264_mp4toannexb.c）
 *   extradata（avcC）→ SPS/PPS（00 00 00 01 头）
 *   样本重写：4/2/1 字节长度前缀 → startcode；IDR 前注入 SPS/PPS
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint8_t *sps; int sps_size; uint8_t *pps; int pps_size;
    int length_size;
    int new_idr, idr_sps_seen, idr_pps_seen, extradata_parsed;
} AnnexB;

#define H264_NAL_SPS 7
#define H264_NAL_PPS 8
#define H264_NAL_SEI 6
#define H264_NAL_IDR_SLICE 5
#define H264_NAL_SLICE 1
#define SEI_TYPE_BUFFERING_PERIOD 0

/* 抄 h264_extradata_to_annexb（avcC → SPS/PPS，startcode 前置） */
int annexb_open(AnnexB *s, const uint8_t *extradata, int extradata_size) {
    uint16_t unit_size;
    uint32_t total_size = 0;
    uint8_t *out = NULL, sps_done = 0;
    static const uint8_t nalu_header[4] = { 0, 0, 0, 1 };
    int length_size, pps_offset = 0;
    const uint8_t *p = extradata, *e = extradata + extradata_size;

    if (extradata_size < 7) return -1;
    p += 4;                                          /* configurationVersion + profile + compat + level */
    length_size = (*p & 0x3) + 1;                    /* 抄 107：lengthSizeMinusOne */
    p++;
    {
        int unit_nb = *p & 0x1f;                     /* 抄 110：numOfSequenceParameterSets */
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
            if (!unit_nb && !sps_done++) {           /* 抄 133-136：PPS 计数 */
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
    if (ps == -1)              start_code_size = 0;   /* PS_OUT_OF_BAND */
    else if (ps == 1 || *out_size == 0) start_code_size = 4;  /* PS_IN_BAND / 首个 */
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

    /* 第一遍：算总长（抄 ffmpeg 双 pass 的 j==0 pass） */
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
        count_or_copy(NULL, &out_size2, buf, (int)nal_size, (unit_type == H264_NAL_SPS || unit_type == H264_NAL_PPS) ? 1 : 0, 0);
        if (unit_type == H264_NAL_SLICE) { new_idr = 1; sps_seen = 0; pps_seen = 0; }
        buf += nal_size;
    }

    *out_buf = malloc(out_size2 ? (size_t)out_size2 : 1);
    if (!*out_buf) return -1;
    /* 第二遍：写（照抄 j==1 pass，状态一致；out_size2 每 pass 重置——
     * 原码 330 行 out_size=0 在 for(j) 循环顶部） */
    out_size2 = 0;
    op = *out_buf;
    buf = buf_in;
    new_idr = s->new_idr; sps_seen = s->idr_sps_seen; pps_seen = s->idr_pps_seen;
    while (buf < buf_end) {
        uint32_t nal_size = 0;
        for (int i = 0; i < s->length_size; i++)
            nal_size = (nal_size << 8) | buf[i];
        buf += s->length_size;
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
        count_or_copy(&op, &out_size2, buf, (int)nal_size, (unit_type == H264_NAL_SPS || unit_type == H264_NAL_PPS) ? 1 : 0, 1);
        if (unit_type == H264_NAL_SLICE) { new_idr = 1; sps_seen = 0; pps_seen = 0; }
        buf += nal_size;
    }
    s->new_idr = new_idr; s->idr_sps_seen = sps_seen; s->idr_pps_seen = pps_seen;
    *out_size = (int)out_size2;
    return 0;
}
