/*
 * shikomi/kaiseki.c — 单路 singlepass 解析器（一次解码全产出：cuts/skeleton/select_frames/events/features；2026-08-23 改名）
 *
 * 用法: ./kaiseki <视频> [<project> <vid_name>] [-o <输出根>]
 * 依赖: shikomi/CUDA_KaKu.cubin（nvcc 编；切点源一字不动 = 内部 CUDA scdet_sad）、
 *       自写 mov 解析 mp4_mov.c、NVCUVID + nvjpeg；video_hash = 文件 SHA256（方案 A）
 * 产物: shikomi/cuts|skeleton|select_frames|events|features/<hash>_*.json + 帧文件（jpg/bin，线 B 写盘）
 *
 * 优化：单次 NVDEC。A 线只做分析 + 在线候选帧缓存；shot 定稿后把少量
 * NV12 候选交给 B 线异步 RGB/224/JPEG + 落盘，写完立即释放候选槽。
 * 不逐帧 JPEG，不第二遍解码；A 不等待 B。
 *
 * 分析链（照抄 unified_extract.c，公式照搬 scdet_vulkan，1341 cuts 为权威）：
 *   切点（scdet_sad 纯 CUDA）/ shot 骨架（在线状态机）/ 在线选帧（process_shot 全套）/
 *   黑帧墙（blackdetect_vulkan 移植）/ quality（yavg/ylow/tout/vrep/brng/entropy）/
 *   sharpness + brightness；静止段/vmafmotion 已裁（2026-08-19 大名定稿：留 切点/黑帧/哈希/选帧）
 *
 * 共享区（A 侧，进程内）：A nvjpeg 编码 jpg + 224 bin → 共享显存环（UE_SHARE_MB 默认 64MB，
 *   SPSC：A 写 tail / B 读 head，B 追着消费即用即丢）；定稿消息内存环形队列（mutex+cond 唤醒 B）；
 *   A 永不等 B：环满/消息满 → overflow 置位后继续（= 帧文件缺 → 整条重跑）；
 *   UE_NO_SHARE=1 → 纯分析单跑（不建共享区不起 B 线程，步骤 1 验收对照用）
 *
 * 编译:
 *   nvcc -arch=all -fatbin -o CUDA_KaKu.cubin CUDA_KaKu.cu
 *   /usr/bin/gcc -O2 -o kaiseki kaiseki.c mp4_mov.c \
 *       -I $CONDA_PREFIX/include -I $CONDA_PREFIX/targets/x86_64-linux/include \
 *       -I <项目>/include/ffnvcodec -lz -lm -ldl -lcuda -lpthread -lcrypto \
 *       -L $CONDA_PREFIX/lib -L /usr/lib/x86_64-linux-gnu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <ctype.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <openssl/sha.h>
#include <cuda.h>
#include <nvjpeg.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include "../include/ffnvcodec/dynlink_nvcuvid.h"   /* NVCUVID（ffmpeg nv-codec-headers） */
#include "hako.h"                        /* 统一容器解析(libavformat 静态, 自包含) + avcC→annexb */

/* ── 参数 ─────────────────────────────────────────────── */
#define SMALL_W         224          /* 评分用缩小帧宽 */
#define SMALL_H         224          /* 评分用缩小帧高 */
#define CENTER_RANGE    15           /* 候选帧取稳定段最低点 ±15 */
#define GRAY_FSZ        (SMALL_H * SMALL_W)        /* 224×224 灰度 50176B（仅 kernel 链中间缓冲，零 DTOH 零落盘） */
#define NB_BLOCKS       ((SMALL_W * SMALL_H + 255) / 256)   /* 小帧 kernel block 数 = 196 */
#define GAUSS_SIGMA     3.0
#define STABLE_PCT      35.0f        /* 稳定段分位阈值 */
#define STILL_MED_THRESH 0.5f        /* 站桩中位数阈值（mafd 域，01 研究标定） */
#define SCDET_THR       10           /* 照抄 00：scdet threshold（分数≥10 切变） */
#define MIN_GAP         2            /* 照抄 00：切变最小帧距 */
#define MAX_CUTS        65536
#define MAX_FRAMES      262144

/* ── S9 黑帧墙（blackdetect_vulkan 移植，参考自 FFmpeg 9.0 源码 vf_blackdetect_vulkan.c）
 *    参数=官方默认（filters.texi）：d=2.0s / pic_th=0.98 / pix_th=0.10 */
#define BLACK_PIC_TH   0.98          /* picture_black_ratio_th 默认 0.98 */
#define BLACK_PIX_TH   0.10          /* pixel_black_th 默认 0.10 */
#define BLACK_DUR_SEC  2.0           /* d 默认 2.0s（最小黑段时长） */
#define BLACK_LUMA_THR ((int)(BLACK_PIX_TH * (235 - 16) + 16))  /* TV range 8bit：16+0.10×219=37.9 → 37（vf_blackdetect_vulkan.c:191-200 换算） */
#define MAX_BLACK      4096
#define MAX_SHOTS      (MAX_CUTS + MAX_BLACK + 8)  /* 在线选帧槽：scdet cut + 黑帧段切点 + 裕量（2026-08-19 大名） */
/* ── 骨架数据结构 ────────────────────────────────────── */
typedef struct { int id, start, end; } Shot;
typedef struct {
    char   video[2048];
    char   video_hash[256];
    double fps;
    int    width, height, total_frames;
    int    n_shots;
    Shot  *shots;
} Skeleton;

static Skeleton sk;

/* ── 全局曲线/帧数据 ─────────────────────────────────── */
static float   *g_mafd = NULL;         /* [total_frames-1] 每帧对 mafd 帧差（=scd_scores 同源，NVOF 退役 01 方案） */
static float   *sharpness = NULL;      /* [total_frames] Laplacian 方差（quality，GPU 算标量） */
static float   *brightness_arr = NULL; /* [total_frames] 亮度有效度（GPU 算标量，选帧直接读） */
static int      work_w, work_h;        /* 解码器输出尺寸 = 全分辨率（偶对齐），分析链/抽帧共用 */

/* ── CUDA scdet（内嵌，与 scdet_vulkan 对齐）───────────── */
static CUmodule   g_mod = 0;
static CUfunction g_fsad = 0;
static CUdeviceptr g_sad_dev = 0;      /* u64 SAD 结果 */
static unsigned long long g_sad = 0;
static double     g_prev_mafd = 0;     /* 上一帧 mafd */
static int        g_cuts[MAX_CUTS];    /* 切变帧号（0 基，= 显示序-1；raw 收集，对拍用） */
static int        g_n_cuts = 0;
static float     *g_scores = NULL;     /* [total_frames] 每帧 scdet 分数（features） */

/* ── S9 黑帧墙（blackdetect_vulkan 移植）───────────────── */
static CUfunction g_fblack = 0;        /* blackdetect_count kernel */
static CUdeviceptr g_black_dev = 0;    /* uint32 黑像素计数 */
static unsigned int g_black = 0;
static int black_start = -1;           /* 首个黑帧号（-1 = NOPTS，照抄 black_start=AV_NOPTS_VALUE） */
static int black_n = 0;                /* 已输出段数 */
static int black_seg_start[MAX_BLACK], black_seg_end[MAX_BLACK];  /* 段 [start, end) 帧号（end=段后第一帧，照抄 end_pts 半开语义） */
/* ── S10 quality（signalstats + entropy 移植，参考自 vf_signalstats.c + vf_entropy.c）── */
static CUfunction g_q_hist = 0, g_q_tout = 0, g_q_vrep = 0, g_q_brng = 0;
static CUdeviceptr g_q_hist_dev = 0;   /* 256×4B Y 直方图 */
static CUdeviceptr g_q_tout_dev = 0, g_q_vrep_dev = 0, g_q_brng_dev = 0;  /* 各 4B 计数 */
static unsigned int g_q_tout_h = 0, g_q_vrep_h = 0, g_q_brng_h = 0;
static float   *g_quality = NULL;      /* [total_frames×9] 每帧: yavg ylow yhigh ymin ymax tout vrep brng entropy */

/* ── GPU 评分小帧（y224_box/y224_lap，零 CPU 像素；灰度 g_small_dev
 *    只作 kernel 链显存中间缓冲，零 DTOH 零落盘）── */
static CUfunction g_ybox = 0, g_ylap = 0;
static CUdeviceptr g_small_dev = 0;    /* 224×224 灰度单帧中转 */
static CUdeviceptr g_bsum_dev = 0;     /* NB_BLOCKS×4B 亮度 block 归约 */
static CUdeviceptr g_lap_dev = 0;      /* NB_BLOCKS×16B laplacian block 归约 */

/* ── 单遍异步产物：A 候选 NV12 → B 编码/写盘 ───────────── */
#define CANDIDATE_CAP   32   /* 每个 shot 最多保留的候选帧；不是最终输出数 */
#define CAND_POOL_CAP   64   /* 全局 GPU NV12 候选池；B 写完立即归还 */
#define B_QUEUE_CAP     64

typedef struct {
    CUdeviceptr dev;
    int state;              /* 0 FREE, 1 ACTIVE(A当前shot), 2 QUEUED(B) */
    int shot_id;
    int fn;
    float rank;             /* 越小越优：以低运动为主 */
} CandidateSlot;

static CandidateSlot g_cand[CAND_POOL_CAP];
static CUdeviceptr g_cand_pool_dev = 0;
static size_t g_nv12_bytes = 0;
static int g_active_cand[CANDIDATE_CAP];
static int g_active_n = 0;

static pthread_mutex_t g_b_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_b_has_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_b_has_free = PTHREAD_COND_INITIALIZER;
static pthread_t g_b_tid;
static volatile int g_b_done = 0;
static int g_b_q[CAND_POOL_CAP];
static int g_b_q_head = 0, g_b_q_n = 0;

/* B 独立编码上下文：A 只负责 candidate D2D copy + queue，不碰 JPEG。 */
static CUstream g_bS = 0;
static nvjpegEncoderState_t g_b_jstate = NULL;
static nvjpegEncoderParams_t g_b_jparams = NULL;
static CUdeviceptr g_b_rgb_dev = 0;
static CUdeviceptr g_b_rgb224_dev = 0;
static unsigned char *g_b_jpg = NULL;
static size_t g_b_jpg_cap = 0;
#define BINSZ           (224 * 224 * 3)   /* 224×224×3 = 150528（DINO 小图 bin 缓冲） */
static unsigned char g_b_bin_buf[BINSZ];
/* ── 输出路径（main 算好） ─────────────────────────────── */
static char g_out_raw[4096], g_out_skel[4096], g_out_sf[4096];
static char g_out_events[4096], g_out_feat[4096];
static char g_vhash[32] = "";          /* 产物文件名前缀 = video_hash 前 6 位（内容指纹，2026-08-19 大名定稿；fnv6 换名实效哈希已禁） */
/* ── 单遍补充声明（编译修复：GPU kernel 指针 / 写盘目录 / 在线选帧 / shot 状态）── */
static CUfunction g_nv12_full = 0;        /* nv12_rgb_full（A 线加载，B 线编码用） */
static CUfunction g_nv12_224 = 0;         /* nv12_rgb224 */
static char     g_frames_dir[4096], g_dino_dir[4096];  /* B 线程写盘目录（main 算好） */
static int *online_reps = NULL;           /* [MAX_SHOTS] 每 shot 代表帧 */
static int **online_kfs = NULL;           /* [MAX_SHOTS] 每 shot 关键帧 */
static int *online_nkfs = NULL;           /* [MAX_SHOTS] 每 shot 关键帧数 */
static int  g_shot_id = 0;                /* shot 编号（scdet cut + 黑帧段切点共用） */
static int  current_shot_start = 0;       /* 当前 open shot 起始帧 */
static MOV g_mov;                      /* S2/S3: 自写 mov 容器（抄 ffmpeg 9.0 mov.c） */

/* 解码会话全局状态（单次解码：decoder_session_open/feed/close 专用） */
static AnnexB  g_ab;
static uint8_t *g_sam = NULL;          /* hako 样本缓冲（64MB） */
static int     g_feed_failed = 0;      /* feed_video_once 失败标记 */

/* 前向声明：在 blackdetect_eval / display_cb 之前使用 */
static void black_seg_split(int seg_start, int seg_end);
static int  q_take_empty(void);
static void q_put_full(int slot, int fn);

/* ── B 线 nvjpeg 句柄（单遍：仅 B 线程使用，b_init_encoder 内 dlopen + dlsym）── */
typedef struct {
    void *lib;
    nvjpegHandle_t jh;
    nvjpegStatus_t (*nvjpegCreateSimple)(nvjpegHandle_t *);
    nvjpegStatus_t (*nvjpegDestroy)(nvjpegHandle_t);
    nvjpegStatus_t (*nvjpegEncoderStateCreate)(nvjpegHandle_t, nvjpegEncoderState_t *, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncoderStateDestroy)(nvjpegEncoderState_t);
    nvjpegStatus_t (*nvjpegEncoderParamsCreate)(nvjpegHandle_t, nvjpegEncoderParams_t *, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncoderParamsDestroy)(nvjpegEncoderParams_t);
    nvjpegStatus_t (*nvjpegEncoderParamsSetQuality)(nvjpegEncoderParams_t, int, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncoderParamsSetSamplingFactors)(nvjpegEncoderParams_t, nvjpegChromaSubsampling_t, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncodeImage)(nvjpegHandle_t, nvjpegEncoderState_t, nvjpegEncoderParams_t,
                                        const nvjpegImage_t *, nvjpegInputFormat_t, int, int, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncodeGetBufferSize)(nvjpegHandle_t, nvjpegEncoderParams_t, int, int, size_t *);
    nvjpegStatus_t (*nvjpegEncodeRetrieveBitstream)(nvjpegHandle_t, nvjpegEncoderState_t,
                                                     unsigned char *, size_t *, cudaStream_t);
} nvjpeg_t;
static nvjpeg_t nj;
#define NJ_LOAD(name) do { *(void **)&nj.name = dlsym(nj.lib, #name); \
    if (!nj.name) die("dlsym " #name " 失败"); } while (0)

/* ═══════════════════════════════════════════════════════
 * 工具
 * ═══════════════════════════════════════════════════════ */
static void die(const char *msg) {
    fprintf(stderr, "unified_extract ERROR: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "unified_extract: malloc %zu bytes failed\n", n); exit(1); }
    return p;
}

static void ck(CUresult r, const char *c) {
    if (r != CUDA_SUCCESS) {
        const char *nm = "?"; cuGetErrorName(r, &nm);
        fprintf(stderr, "unified_extract CUDA %s failed: %s (%d)\n", c, nm, (int)r);
        exit(1);
    }
}

/* JPEG 质量：默认 60（领导定档 2026-08-06）；环境变量 FULLRES_JPG_QUALITY 可覆盖（照抄现役） */
static int jpg_quality(void) {
    const char *e = getenv("FULLRES_JPG_QUALITY");
    if (e) { int q = atoi(e); if (q >= 1 && q <= 100) return q; }
    return 60;
}

static char g_script_dir[1024] = ".";

static void get_script_dir(void) {
    ssize_t len = readlink("/proc/self/exe", g_script_dir, sizeof(g_script_dir) - 1);
    if (len <= 0) { snprintf(g_script_dir, sizeof(g_script_dir), "."); return; }
    g_script_dir[len] = 0;
    char *slash = strrchr(g_script_dir, '/');
    if (slash) *slash = 0;
}

static void gpu_load_cubin(const char *name) {
    char cubin_path[1024];
    snprintf(cubin_path, sizeof(cubin_path), "%s/%s", g_script_dir, name);
    FILE *f = fopen(cubin_path, "rb");
    if (!f) die("打开 cubin 失败（先 nvcc -arch=all -fatbin）");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = xmalloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) {}
    fclose(f);
    ck(cuModuleLoadDataEx(&g_mod, buf, 0, NULL, NULL), "load cubin");
    free(buf);
}

static void gpu_get_fn(const char *fn_name, CUfunction *out) {
    ck(cuModuleGetFunction(out, g_mod, fn_name), "get function");
}

/* nvjpeg 仅在 B 线程使用（b_init_encoder 内 dlopen + 建 encoder state/params），A 线不需要。
 * 旧双线 nvjpeg_open() 已移除，避免重复 dlopen 与 nj 结构体残留。 */

static void mkdir_p(const char *path) {
    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    mkdir(tmp, 0755);
}

/* 原子写帧文件：第二遍选择性编码专用。 */
static void write_frame_file(const char *dir, const char *name, const void *data, size_t len) {
    char tmp[8192], fin[8192];
    snprintf(fin, sizeof(fin), "%s/%s", dir, name);
    if (access(fin, F_OK) == 0) return;
    snprintf(tmp, sizeof(tmp), "%s/.%s.tmp", dir, name);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "fopen %s 失败: %s\n", tmp, strerror(errno)); return; }
    if (fwrite(data, 1, len, f) != len) {
        fclose(f); unlink(tmp);
        fprintf(stderr, "fwrite %s 失败\n", tmp);
        return;
    }
    fclose(f);
    if (rename(tmp, fin) != 0) {
        fprintf(stderr, "rename %s 失败: %s\n", tmp, strerror(errno));
        unlink(tmp);
    }
}

static void mkdir_for(const char *file_path) {
    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s", file_path);
    char *slash = strrchr(tmp, '/');
    if (slash) {
        *slash = 0;
        char *p = tmp + 1;
        for (; *p; p++)
            if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
        mkdir(tmp, 0755);
    }
}

/* ── FNV-1a 32 位（视频名 → 6 位 hex 前缀；下游 Python 同算法，2026-08-06 领导铁律）── */
static unsigned fnv1a32(const char *s) {
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

/* ═══════════════════════════════════════════════════════
 * CUDA 上下文（异步：解码线程 = 拷贝入队，处理线程 = 分析+选帧+共享产物）
 * ═══════════════════════════════════════════════════════ */
static CUcontext       ctx = NULL;
static CUstream        procS = 0;       /* 第一遍分析流 */
#define RING_CAP        64              /* 帧环形缓冲（背压上限；1080p NV12 ≈ 200MB） */
static CUdeviceptr     dIn[RING_CAP];   /* 帧环形缓冲（display 拷贝入队，处理线程消费） */
static uint32_t        inPitch, inUVoff;
static int             g_prev_slot = -1; /* 处理线程：上一帧 slot（scdet 帧差用，占 1 槽） */

/* 帧队列（生产=display_cb，消费=处理线程；满则背压阻塞解码） */
static pthread_mutex_t g_q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_q_has_data = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_q_has_slot = PTHREAD_COND_INITIALIZER;
static int g_q_slot[RING_CAP], g_q_fn[RING_CAP];
static int g_q_head = 0, g_q_n = 0;
static int g_empty_stack[RING_CAP];     /* 空闲 slot 栈 */
static int g_empty_n = 0;
static volatile int g_eos = 0;          /* 解码结束标志（处理队尾后收尾） */
static volatile int g_proc_done = 0;    /* 处理线程已完全结束（B 写盘线程放行信号） */
static pthread_t g_proc_tid;            /* 处理线程（分析+选帧+保存） */

static void cuda_init(int w, int h) {
    ck(cuInit(0), "cuInit");
    CUdevice dev; ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
    ck(cuCtxCreate(&ctx, NULL, 0, dev), "cuCtxCreate");
    ck(cuStreamCreate(&procS, CU_STREAM_NON_BLOCKING), "procStream");

    /* 帧环形缓冲（NV12 布局：Y 平面 + UV 交错） */
    inPitch  = (uint32_t)(((size_t)w + 127) & ~(size_t)127);   /* 2D 拷贝行对齐 */
    inUVoff  = inPitch * (uint32_t)h;
    size_t nv12 = (size_t)inUVoff + (size_t)inPitch * ((uint32_t)h / 2);
    for (int i = 0; i < RING_CAP; i++) {
        char nm[32]; snprintf(nm, sizeof(nm), "alloc dIn%d", i);
        ck(cuMemAlloc(&dIn[i], nv12), nm);
        g_empty_stack[g_empty_n++] = i;
    }
}

/* 上传一帧 NV12（Y 平面 + UV 交错平面）——显存源（NVCUVID 解码帧）GPU→GPU，零 CPU 往返 */
static void upload_nv12_dev(CUdeviceptr src, size_t srcPitch, CUdeviceptr dst) {
    CUDA_MEMCPY2D c; memset(&c, 0, sizeof(c));
    c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    c.dstDevice = dst; c.dstPitch = inPitch;
    c.srcDevice = src; c.srcPitch = srcPitch;
    c.WidthInBytes = (size_t)work_w;
    c.Height = (size_t)work_h;
    ck(cuMemcpy2D(&c), "devcpy Y");   /* 同步拷贝（默认流）：与 NVDEC 后处理写表面隐式同步 */
    c.srcDevice = src + srcPitch * work_h;   /* map 返回 target 尺寸表面（pitch 已按 128 对齐），UV 起点 = pitch × target 高 */
    c.Height = (size_t)work_h / 2;
    c.dstDevice = dst + inUVoff;
    ck(cuMemcpy2D(&c), "devcpy UV");
}

/* scdet_sad kernel：Y 平面帧对 SAD（与 scdet_vulkan 逐位对齐，整数运算） */
static void scdet_sad_kernel(CUdeviceptr prev, CUdeviceptr cur, uint32_t pitch) {
    ck(cuMemsetD8Async(g_sad_dev, 0, 8, procS), "zero sad");
    unsigned long long a1 = (unsigned long long)prev, a2 = (unsigned long long)cur;
    unsigned long long a3 = (unsigned long long)g_sad_dev;
    int aP = (int)pitch, aW = work_w, aH = work_h;
    void *kp[6] = { &a1, &a2, &aP, &a3, &aW, &aH };
    ck(cuLaunchKernel(g_fsad, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "scdet_sad");
}

static void blackdetect_count_kernel(CUdeviceptr y, uint32_t pitch) {
    ck(cuMemsetD8Async(g_black_dev, 0, 4, procS), "zero black");
    unsigned long long a1 = (unsigned long long)y;
    int aP = (int)pitch, aW = work_w, aH = work_h, aT = BLACK_LUMA_THR;
    unsigned long long a6 = (unsigned long long)g_black_dev;
    void *kp[6] = { &a1, &aP, &aW, &aH, &aT, &a6 };
    ck(cuLaunchKernel(g_fblack, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "blackdetect");
}

static void q_hist_kernel(CUdeviceptr y, uint32_t pitch) {
    ck(cuMemsetD8Async(g_q_hist_dev, 0, 256 * 4, procS), "zero qhist");
    unsigned long long a1 = (unsigned long long)y, a6 = (unsigned long long)g_q_hist_dev;
    int aP = (int)pitch, aW = work_w, aH = work_h;
    void *kp[5] = { &a1, &aP, &aW, &aH, &a6 };
    ck(cuLaunchKernel(g_q_hist, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "q_hist");
}
static void q_tout_kernel(CUdeviceptr y, uint32_t pitch) {
    ck(cuMemsetD8Async(g_q_tout_dev, 0, 4, procS), "zero qtout");
    unsigned long long a1 = (unsigned long long)y, a6 = (unsigned long long)g_q_tout_dev;
    int aP = (int)pitch, aW = work_w, aH = work_h;
    void *kp[5] = { &a1, &aP, &aW, &aH, &a6 };
    ck(cuLaunchKernel(g_q_tout, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "q_tout");
}
static void q_vrep_kernel(CUdeviceptr y, uint32_t pitch) {
    ck(cuMemsetD8Async(g_q_vrep_dev, 0, 4, procS), "zero qvrep");
    unsigned long long a1 = (unsigned long long)y, a6 = (unsigned long long)g_q_vrep_dev;
    int aP = (int)pitch, aW = work_w, aH = work_h;
    void *kp[5] = { &a1, &aP, &aW, &aH, &a6 };
    ck(cuLaunchKernel(g_q_vrep, (work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "q_vrep");
}
static void q_brng_kernel(CUdeviceptr y, CUdeviceptr uv, uint32_t pitch) {
    ck(cuMemsetD8Async(g_q_brng_dev, 0, 4, procS), "zero qbrng");
    unsigned long long a1 = (unsigned long long)y, a2 = (unsigned long long)uv, a6 = (unsigned long long)g_q_brng_dev;
    int aP = (int)pitch, aW = work_w, aH = work_h;
    void *kp[6] = { &a1, &aP, &a2, &aW, &aH, &a6 };
    ck(cuLaunchKernel(g_q_brng, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "q_brng");
}

/* S12 CPU 求值（照抄 vf_signalstats.c SET_META） */
static void quality_eval(int frame, const uint32_t *hist) {
    double fs = (double)work_w * work_h;
    long tot = 0;
    int ymin = -1, ymax = -1;
    for (int v = 0; v < 256; v++) {
        if (hist[v]) { if (ymin < 0) ymin = v; ymax = v; }
        tot += (long)hist[v] * v;
    }
    long cum = 0, llimit = lrint(fs * 10 / 100), hlimit = lrint(fs * 90 / 100);
    int ylow = -1, yhigh = -1;
    for (int v = 0; v < 256; v++) {
        cum += hist[v];
        if (ylow < 0 && cum >= llimit) ylow = v;
        if (yhigh < 0 && cum >= hlimit) yhigh = v;
    }
    float total = (float)(work_w * work_h);
    float entropy = 0.0f;
    for (int v = 0; v < 256; v++)
        if (hist[v]) {
            float p = (float)hist[v] / total;
            entropy += (float)(-log2((double)p) * (double)p);
        }
    float *q = &g_quality[(size_t)frame * 9];
    q[0] = (float)((double)tot / fs);
    q[1] = (float)ylow;  q[2] = (float)yhigh;
    q[3] = (float)ymin;  q[4] = (float)ymax;
    q[5] = (float)((double)g_q_tout_h / fs);
    q[6] = (float)((double)g_q_vrep_h * work_w / fs);
    q[7] = (float)((double)g_q_brng_h / fs);
    q[8] = (float)entropy;
}

/* S9 宿主状态机（照抄 vf_blackdetect_vulkan.c:evaluate + report_black_region） */
static void blackdetect_eval(int frame) {
    double ratio = (double)g_black / ((double)work_w * work_h);
    if (ratio >= BLACK_PIC_TH) {
        if (black_start < 0) black_start = frame;
    } else if (black_start >= 0) {
        if ((frame - black_start) >= (long long)(BLACK_DUR_SEC * sk.fps)) {
            if (black_n < MAX_BLACK) {   /* 黑帧墙记录（照抄 report_black_region） */
                black_seg_start[black_n] = black_start;
                black_seg_end[black_n] = frame;
                black_n++;
            }
            black_seg_split(black_start, frame - 1);   /* 黑帧段独立 shot（2026-08-19 大名） */
        }
        black_start = -1;
    }
}

static void cuda_destroy(void) {
    if (g_cand_pool_dev) { ck(cuMemFree(g_cand_pool_dev), "free candidate pool"); g_cand_pool_dev = 0; }
    if (g_sad_dev) ck(cuMemFree(g_sad_dev), "free sad");
    if (g_black_dev) ck(cuMemFree(g_black_dev), "free black");
    if (g_q_brng_dev) ck(cuMemFree(g_q_brng_dev), "free q_brng");
    if (g_q_vrep_dev) ck(cuMemFree(g_q_vrep_dev), "free q_vrep");
    if (g_q_tout_dev) ck(cuMemFree(g_q_tout_dev), "free q_tout");
    if (g_q_hist_dev) ck(cuMemFree(g_q_hist_dev), "free q_hist");
    if (g_lap_dev) ck(cuMemFree(g_lap_dev), "free lap");
    if (g_bsum_dev) ck(cuMemFree(g_bsum_dev), "free bsum");
    if (g_small_dev) ck(cuMemFree(g_small_dev), "free small");
    for (int i = 0; i < RING_CAP; i++) if (dIn[i]) ck(cuMemFree(dIn[i]), "free dIn-ring");
    if (procS) ck(cuStreamDestroy(procS), "destroy procS");
    if (g_mod) ck(cuModuleUnload(g_mod), "unload module");
    if (ctx) ck(cuCtxDestroy(ctx), "destroy ctx");
}


/* ═══════════════════════════════════════════════════════
 * NVCUVID 解码（显存直出，零 CPU 帧搬运；输出全分辨率）
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    void *lib;
    tcuvidCreateVideoParser  *cuvidCreateVideoParser;
    tcuvidParseVideoData     *cuvidParseVideoData;
    tcuvidDestroyVideoParser *cuvidDestroyVideoParser;
    tcuvidCreateDecoder      *cuvidCreateDecoder;
    tcuvidDestroyDecoder     *cuvidDestroyDecoder;
    tcuvidDecodePicture      *cuvidDecodePicture;
    tcuvidMapVideoFrame      *cuvidMapVideoFrame;
    tcuvidUnmapVideoFrame    *cuvidUnmapVideoFrame;
} NVCUVID;

static NVCUVID nv;
static CUvideoparser  g_parser  = NULL;
static CUvideodecoder g_decoder = NULL;
static int g_disp_count = 0;      /* 显示序帧计数 */

#define NVCUVID_LOAD(name) do { *(void **)&nv.name = dlsym(nv.lib, #name); if (!nv.name) die("dlsym " #name " 失败"); } while (0)

static void *nvcuvid_sym(const char *n1, const char *n2) {
    void *p = dlsym(nv.lib, n1);
    if (!p && n2) p = dlsym(nv.lib, n2);
    return p;
}

static int sequence_cb(void *user, CUVIDEOFORMAT *fmt);
static int decode_cb(void *user, CUVIDPICPARAMS *pic);
static int display_cb(void *user, CUVIDPARSERDISPINFO *disp);

static void nvcuvid_open(void) {
    memset(&nv, 0, sizeof(nv));
    nv.lib = dlopen("libnvcuvid.so.1", RTLD_LAZY);
    if (!nv.lib) die("dlopen libnvcuvid.so.1 失败（驱动是否安装 NVCUVID？）");
    NVCUVID_LOAD(cuvidCreateVideoParser);
    NVCUVID_LOAD(cuvidParseVideoData);
    NVCUVID_LOAD(cuvidDestroyVideoParser);
    NVCUVID_LOAD(cuvidCreateDecoder);
    NVCUVID_LOAD(cuvidDestroyDecoder);
    NVCUVID_LOAD(cuvidDecodePicture);
    *(void **)&nv.cuvidMapVideoFrame   = nvcuvid_sym("cuvidMapVideoFrame64",   "cuvidMapVideoFrame");
    *(void **)&nv.cuvidUnmapVideoFrame = nvcuvid_sym("cuvidUnmapVideoFrame64", "cuvidUnmapVideoFrame");
    if (!nv.cuvidMapVideoFrame || !nv.cuvidUnmapVideoFrame) die("dlsym cuvidMap/Unmap 失败");

}

/* sequence 回调：码流格式就绪 → 创建解码器（输出全分辨率，偶对齐） */
static int sequence_cb(void *user, CUVIDEOFORMAT *fmt) {
    (void)user;
    if (g_decoder) return 1;
    CUVIDDECODECREATEINFO info; memset(&info, 0, sizeof(info));
    info.ulWidth          = fmt->coded_width;
    info.ulHeight         = fmt->coded_height;
    info.ulTargetWidth    = (unsigned int)work_w;   /* 全分辨率 */
    info.ulTargetHeight   = (unsigned int)work_h;
    info.bitDepthMinus8   = fmt->bit_depth_luma_minus8;
    info.OutputFormat     = cudaVideoSurfaceFormat_NV12;
    info.CodecType        = fmt->codec;
    info.ChromaFormat     = fmt->chroma_format;
    info.DeinterlaceMode  = cudaVideoDeinterlaceMode_Weave;
    info.ulNumDecodeSurfaces = 20;
    info.ulNumOutputSurfaces = 4;
    CUresult cr = nv.cuvidCreateDecoder(&g_decoder, &info);
    if (cr != CUDA_SUCCESS) {
        const char *nm = "?";
        cuGetErrorName(cr, &nm);
        fprintf(stderr, "unified_extract: cuvidCreateDecoder 失败 (%dx%d -> %dx%d): %s (%d)\n",
                (int)info.ulWidth, (int)info.ulHeight, work_w, work_h, nm, (int)cr);
        return 0;
    }
    printf("  解码器: %dx%d -> %dx%d (NVDEC 全分辨率)\n",
           (int)fmt->coded_width, (int)fmt->coded_height, work_w, work_h);
    return 1;
}

static int decode_cb(void *user, CUVIDPICPARAMS *pic) {
    (void)user;
    if (!g_decoder) return 0;
    CUresult r = nv.cuvidDecodePicture(g_decoder, pic);
    if (r != CUDA_SUCCESS) {
        const char *nm = "?";
        cuGetErrorName(r, &nm);
        fprintf(stderr, "unified_extract: cuvidDecodePicture 失败 pic=%d: %s (%d)\n", pic->CurrPicIdx, nm, (int)r);
        return 0;
    }
    return 1;
}

/* display 回调：解码器 NV12 表面就绪 → 拷贝到环形缓冲 slot → 通知处理线程 */
static int display_cb(void *user, CUVIDPARSERDISPINFO *disp) {
    (void)user;
    if (!g_decoder) return 0;
    unsigned long long dptr64 = 0;
    unsigned int pitch = 0;
    CUVIDPROCPARAMS vpp; memset(&vpp, 0, sizeof(vpp));
    vpp.progressive_frame = disp->progressive_frame;
    vpp.second_field      = disp->repeat_first_field;
    vpp.top_field_first   = disp->top_field_first;
    vpp.unpaired_field    = disp->repeat_first_field;
    ck(cuCtxPushCurrent(ctx), "push ctx");
    CUresult mr = nv.cuvidMapVideoFrame(g_decoder, disp->picture_index, &dptr64, &pitch, &vpp);
    if (mr != CUDA_SUCCESS) {
        const char *nm = "?";
        cuGetErrorName(mr, &nm);
        cuCtxPopCurrent(NULL);
        fprintf(stderr, "unified_extract: cuvidMapVideoFrame 失败 pic=%d: %s (%d)\n", disp->picture_index, nm, (int)mr);
        return 0;
    }
    CUdeviceptr dptr = (CUdeviceptr)dptr64;
    int i = g_disp_count++;
    if (i && i % 10000 == 0) { fprintf(stderr, "[unified_extract] 解码中 %d 帧\n", i); fflush(stderr); }
    if (i >= sk.total_frames) {   /* 越界保护 */
        nv.cuvidUnmapVideoFrame(g_decoder, dptr64);
        cuCtxPopCurrent(NULL);
        return 1;
    }

    int slot = q_take_empty();
    upload_nv12_dev(dptr, (size_t)pitch, dIn[slot]);
    ck(cuStreamSynchronize(procS), "sync in");
    nv.cuvidUnmapVideoFrame(g_decoder, dptr64);
    cuCtxPopCurrent(NULL);
    q_put_full(slot, i);
    return 1;
}

/* ═══════════════════════════════════════════════════════
 * 稳定段检测（照抄 Preproc）
 * ═══════════════════════════════════════════════════════ */
typedef struct { int start, end; } Region;

static void gaussian_smooth(const float *in, int n, float *out) {
    int r = (int)ceil(GAUSS_SIGMA * 4);
    float *kernel = xmalloc(sizeof(float) * (2 * r + 1));
    float sum = 0;
    for (int i = -r; i <= r; i++) {
        kernel[i + r] = expf(-(i * i) / (2 * GAUSS_SIGMA * GAUSS_SIGMA));
        sum += kernel[i + r];
    }
    for (int i = -r; i <= r; i++) kernel[i + r] /= sum;
    for (int i = 0; i < n; i++) {
        float acc = 0;
        for (int j = -r; j <= r; j++) {
            int k = i + j;
            if (k < 0) k = 0; else if (k >= n) k = n - 1;
            acc += in[k] * kernel[j + r];
        }
        out[i] = acc;
    }
    free(kernel);
}

static int cmp_int(const void *a, const void *b) {
    return *(const int *)a - *(const int *)b;
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static Region *detect_stable_regions(const float *seg, int n, int min_stable, int *out_n) {
    Region *regions;
    if (n < 2) { *out_n = 1; regions = xmalloc(sizeof(Region)); regions[0].start = 0; regions[0].end = n - 1; return regions; }

    float *smooth = xmalloc(sizeof(float) * n);
    gaussian_smooth(seg, n, smooth);

    float *tmp = xmalloc(sizeof(float) * n);
    memcpy(tmp, smooth, sizeof(float) * n);
    qsort(tmp, n, sizeof(float), cmp_float);
    float threshold = tmp[(int)(STABLE_PCT * n / 100.0f)];
    free(tmp);

    int cap = n / 2 + 2;
    regions = xmalloc(sizeof(Region) * cap);
    int nr = 0, i = 0;
    while (i < n) {
        if (smooth[i] < threshold) {
            int s = i;
            while (i < n && smooth[i] < threshold) i++;
            int e = i - 1;
            if (e - s + 1 >= min_stable && nr < cap)
                regions[nr++] = (Region){s, e};
        } else i++;
    }
    if (nr == 0) {
        float bestv = seg[0]; int best = 0;
        for (int k = 1; k < n; k++) if (seg[k] < bestv) { bestv = seg[k]; best = k; }
        int half = min_stable / 2; if (half < 1) half = 1;
        int s = best - half, e = best + half;
        if (s < 0) s = 0; if (e >= n) e = n - 1;
        regions[0] = (Region){s, e}; nr = 1;
    }

    {
        int MERGE_GAP = 3;
        int i = 0;
        while (i + 1 < nr) {
            if (regions[i + 1].start - regions[i].end <= MERGE_GAP) {
                regions[i].end = regions[i + 1].end;
                for (int k = i + 1; k < nr - 1; k++) regions[k] = regions[k + 1];
                nr--;
            } else i++;
        }
    }
    free(smooth);
    *out_n = nr;
    return regions;
}

typedef struct { int fn; float score; } KF;

/* 单镜头处理（照抄 Preproc；选帧输入 g_mafd/sharpness/g_quality/brightness_arr 解码期间已全量驻留） */
static void process_shot(const Shot *shot, int *rep_out, int **kfs_out, int *n_kfs_out) {
    int seg_start = shot->start, seg_end = shot->end;
    int n_frames = seg_end - seg_start + 1;
    double dur_s = n_frames / (sk.fps > 0 ? sk.fps : 1);
    int min_stable = (int)(0.5 * sk.fps); if (min_stable < 1) min_stable = 1;
    int max_kf;
    if (dur_s < 2) max_kf = 2; else if (dur_s < 30) max_kf = 4; else max_kf = 8;

    int seg_len = seg_end - seg_start;
    if (seg_len < 1) seg_len = 1;
    float *seg = xmalloc(sizeof(float) * seg_len);
    for (int i = 0; i < seg_len; i++) {
        int idx = seg_start + i;
        seg[i] = (idx < sk.total_frames - 1) ? g_mafd[idx] : 0.0f;
    }

    int n_regions = 0;
    Region *regions = detect_stable_regions(seg, seg_len, min_stable, &n_regions);
    if (n_regions > max_kf) n_regions = max_kf;

    /* 站桩特判：shot 整体运动极低 → 只保留一个稳定段，避免重复抽帧 */
    {
        float *tmp = xmalloc(sizeof(float) * seg_len);
        memcpy(tmp, seg, sizeof(float) * seg_len);
        qsort(tmp, seg_len, sizeof(float), cmp_float);
        float med = tmp[seg_len / 2];
        free(tmp);
        if (med < STILL_MED_THRESH && n_regions > 1) {
            int gbest = 0; float bv = 1e30f;
            for (int i = 0; i < seg_len; i++) if (seg[i] < bv) { bv = seg[i]; gbest = i; }
            int matched = 0;
            for (int r = 0; r < n_regions; r++)
                if (gbest >= regions[r].start && gbest <= regions[r].end) {
                    regions[0] = regions[r]; n_regions = 1; matched = 1; break;
                }
            if (!matched) {
                int best_r = 0, best_len = regions[0].end - regions[0].start;
                for (int r = 1; r < n_regions; r++) {
                    int len = regions[r].end - regions[r].start;
                    if (len > best_len) { best_len = len; best_r = r; }
                }
                regions[0] = regions[best_r]; n_regions = 1;
            }
        }
    }

    KF *kfs = xmalloc(sizeof(KF) * (n_regions + 8));
    int n_kfs = 0;

    for (int ri = 0; ri < n_regions; ri++) {
        int rel_sf = regions[ri].start, rel_ef = regions[ri].end;
        int zone_len = rel_ef - rel_sf + 1;
        if (zone_len < 1) continue;

        int best_rel_in_zone = 0;
        float bestv = 1e30f;
        for (int i = 0; i < zone_len; i++) {
            float v = seg[rel_sf + i];
            if (v < bestv) { bestv = v; best_rel_in_zone = i; }
        }
        int best_abs = seg_start + rel_sf + best_rel_in_zone;

        int cand[2 * CENTER_RANGE + 1]; int nc = 0;
        for (int off = -CENTER_RANGE; off <= CENTER_RANGE; off++) {
            int cf = best_rel_in_zone + off;
            if (cf >= 0 && cf < zone_len) cand[nc++] = seg_start + rel_sf + cf;
        }
        if (nc == 0) { kfs[n_kfs++] = (KF){best_abs, 0.0f}; continue; }

        float stab_max = 1e-9f;
        for (int i = 0; i < zone_len; i++) if (seg[rel_sf + i] > stab_max) stab_max = seg[rel_sf + i];

        /* #30 features 质量分：锐度 sharpness / 熵 g_quality[fn*9+8] / 过曝 TOUT g_quality[fn*9+5] */
        float sharp_max = 0, ent_max = 0;
        for (int ci = 0; ci < nc; ci++) {
            float sv = sharpness[cand[ci]]; if (sv > sharp_max) sharp_max = sv;
            float ev = g_quality[(size_t)cand[ci] * 9 + 8]; if (ev > ent_max) ent_max = ev;
        }
        if (sharp_max < 1e-9f) sharp_max = 1.0f;
        if (ent_max < 1e-9f) ent_max = 1.0f;

        int best_ci = 0; float best_score = -1e30f;
        for (int ci = 0; ci < nc; ci++) {
            int fn = cand[ci];
            float motion_v = seg[fn - seg_start];
            if (motion_v > stab_max) motion_v = stab_max;
            float stab = 1.0f - motion_v / stab_max;
            float sharp = sharpness[fn] / sharp_max;
            float ent   = g_quality[(size_t)fn * 9 + 8] / ent_max;
            float tout  = g_quality[(size_t)fn * 9 + 5];
            float bright = brightness_arr[fn];

            float score = 0.4f * stab + 0.25f * sharp + 0.2f * ent + 0.15f * bright;
            if (tout > 0.05f) score *= (1.0f - (tout - 0.05f) / 0.95f);
            if (score > best_score) { best_score = score; best_ci = ci; }
        }
        kfs[n_kfs++] = (KF){cand[best_ci], best_score};
    }
    free(regions); free(seg);

    /* 不去重：候选帧直接输出（2026-08-07 领导：去重是 DINO 聚类之后的事） */

    for (int i = 1; i < n_kfs; i++) {
        KF v = kfs[i]; int j = i - 1;
        while (j >= 0 && kfs[j].fn > v.fn) { kfs[j + 1] = kfs[j]; j--; }
        kfs[j + 1] = v;
    }

    int rep = kfs[0].fn; float best = kfs[0].score;
    for (int i = 1; i < n_kfs; i++)
        if (kfs[i].score > best) { best = kfs[i].score; rep = kfs[i].fn; }
    if (n_kfs == 0) rep = (seg_start + seg_end) / 2;

    int *out = xmalloc(sizeof(int) * (n_kfs ? n_kfs : 1));
    for (int i = 0; i < n_kfs; i++) out[i] = kfs[i].fn;
    free(kfs);

    *rep_out = rep;
    *kfs_out = out;
    *n_kfs_out = n_kfs;
}

/* ── 候选帧池 / B 写盘线程 ───────────────────────────── */
static void candidate_pool_init(void) {
    ck(cuMemAlloc(&g_cand_pool_dev, g_nv12_bytes * CAND_POOL_CAP), "alloc candidate pool");
    memset(g_cand, 0, sizeof(g_cand));
    for (int i = 0; i < CAND_POOL_CAP; i++)
        g_cand[i].dev = g_cand_pool_dev + (size_t)i * g_nv12_bytes;
    g_active_n = 0;
}

static int candidate_find_exact(int fn) {
    for (int i = 0; i < g_active_n; i++)
        if (g_cand[g_active_cand[i]].fn == fn) return g_active_cand[i];
    return -1;
}

static int candidate_find_near(int fn, int used_mask) {
    int best = -1, best_d = 1 << 30;
    for (int i = 0; i < g_active_n; i++) {
        int si = g_active_cand[i];
        if (used_mask & (1u << (i & 31))) continue;
        int d = abs(g_cand[si].fn - fn);
        if (d < best_d) { best_d = d; best = si; }
    }
    return best;
}

static int candidate_free_slot(void) {
    pthread_mutex_lock(&g_b_mu);
    for (int i = 0; i < CAND_POOL_CAP; i++) {
        if (g_cand[i].state == 0) {
            g_cand[i].state = 1;
            pthread_mutex_unlock(&g_b_mu);
            return i;
        }
    }
    pthread_mutex_unlock(&g_b_mu);
    return -1;
}

/* 当前帧的低运动优先级；最终质量仍由 process_shot 决定。 */
static float candidate_rank(int fn) {
    float motion = (fn > 0) ? g_mafd[fn - 1] : 0.0f;
    /* 以 motion 为主，轻量惩罚明显过曝/极暗；避免候选池被黑帧占满。 */
    float tout = g_quality[(size_t)fn * 9 + 5];
    float bright = brightness_arr[fn];
    float penalty = 0.0f;
    if (tout > 0.05f) penalty += (tout - 0.05f) * 8.0f;
    penalty += (1.0f - bright) * 0.5f;
    return motion + penalty;
}

static void candidate_copy_from_frame(int slot_idx, int src_slot, int fn, int shot_id, float rank) {
    CandidateSlot *c = &g_cand[slot_idx];
    ck(cuMemcpyDtoDAsync(c->dev, dIn[src_slot], g_nv12_bytes, procS), "candidate D2D");
    c->fn = fn;
    c->shot_id = shot_id;
    c->rank = rank;
}

/* A 每帧最多保留 32 个低运动/非黑候选；只在候选升级时做一次 GPU D2D。 */
static void candidate_consider(int fn, int src_slot, int shot_id) {
    float rank = candidate_rank(fn);

    /* 与已有候选过近时，只保留更优的一个，形成时间上的分散候选。 */
    for (int i = 0; i < g_active_n; i++) {
        int si = g_active_cand[i];
        if (g_cand[si].shot_id != shot_id) continue;
        if (abs(g_cand[si].fn - fn) <= CENTER_RANGE) {
            if (rank < g_cand[si].rank)
                candidate_copy_from_frame(si, src_slot, fn, shot_id, rank);
            return;
        }
    }

    if (g_active_n < CANDIDATE_CAP) {
        int si = candidate_free_slot();
        if (si < 0) return; /* B 短时积压：A 不等待，放弃这个次级候选 */
        g_active_cand[g_active_n++] = si;
        candidate_copy_from_frame(si, src_slot, fn, shot_id, rank);
        return;
    }

    int worst_i = -1;
    float worst = -1.0f;
    for (int i = 0; i < g_active_n; i++) {
        int si = g_active_cand[i];
        if (g_cand[si].rank > worst) { worst = g_cand[si].rank; worst_i = i; }
    }
    if (worst_i >= 0 && rank < worst) {
        int si = g_active_cand[worst_i];
        candidate_copy_from_frame(si, src_slot, fn, shot_id, rank);
    }
}

static int candidate_choose_for_frame(int fn, unsigned used_mask) {
    int exact = candidate_find_exact(fn);
    if (exact >= 0) return exact;
    int best = candidate_find_near(fn, used_mask);
    return best;
}

static void b_release_slot(int si) {
    pthread_mutex_lock(&g_b_mu);
    g_cand[si].state = 0;
    pthread_cond_signal(&g_b_has_free);
    pthread_mutex_unlock(&g_b_mu);
}

static void b_enqueue_slot(int si) {
    pthread_mutex_lock(&g_b_mu);
    if (g_b_q_n >= B_QUEUE_CAP) {
        pthread_mutex_unlock(&g_b_mu);
        fprintf(stderr, "extract_thread: B queue full，丢弃 frame=%d\n", g_cand[si].fn);
        b_release_slot(si);
        return;
    }
    g_cand[si].state = 2;
    int at = (g_b_q_head + g_b_q_n) % B_QUEUE_CAP;
    g_b_q[at] = si;
    g_b_q_n++;
    pthread_cond_signal(&g_b_has_work);
    pthread_mutex_unlock(&g_b_mu);
}

static int b_dequeue_slot(void) {
    pthread_mutex_lock(&g_b_mu);
    while (g_b_q_n == 0 && !g_b_done)
        pthread_cond_wait(&g_b_has_work, &g_b_mu);
    if (g_b_q_n == 0 && g_b_done) {
        pthread_mutex_unlock(&g_b_mu);
        return -1;
    }
    int si = g_b_q[g_b_q_head];
    g_b_q_head = (g_b_q_head + 1) % B_QUEUE_CAP;
    g_b_q_n--;
    pthread_mutex_unlock(&g_b_mu);
    return si;
}

static void b_init_encoder(void) {
    /* 仅 B 线程 dlopen libnvjpeg + dlsym（旧双线 nvjpeg_open 已并入此处） */
    memset(&nj, 0, sizeof(nj));
    nj.lib = dlopen("libnvjpeg.so.13", RTLD_LAZY);
    if (!nj.lib) {
        const char *cp = getenv("CONDA_PREFIX");
        const char *home = getenv("HOME");
        char p[1024];
        const char *tries[] = {
            "envs/amaterasu/lib/libnvjpeg.so.13",
            "lib/libnvjpeg.so.13",
        };
        for (size_t i = 0; i < sizeof(tries) / sizeof(tries[0]) && !nj.lib; i++) {
            if (cp) {
                snprintf(p, sizeof(p), "%s/%s", cp, tries[i]);
                nj.lib = dlopen(p, RTLD_LAZY);
            }
            if (!nj.lib && home) {
                snprintf(p, sizeof(p), "%s/miniconda3/%s", home, tries[i]);
                nj.lib = dlopen(p, RTLD_LAZY);
            }
        }
    }
    if (!nj.lib) die("dlopen libnvjpeg.so.13 失败");
    NJ_LOAD(nvjpegCreateSimple);
    NJ_LOAD(nvjpegDestroy);
    NJ_LOAD(nvjpegEncoderStateCreate);
    NJ_LOAD(nvjpegEncoderStateDestroy);
    NJ_LOAD(nvjpegEncoderParamsCreate);
    NJ_LOAD(nvjpegEncoderParamsDestroy);
    NJ_LOAD(nvjpegEncoderParamsSetQuality);
    NJ_LOAD(nvjpegEncoderParamsSetSamplingFactors);
    NJ_LOAD(nvjpegEncodeImage);
    NJ_LOAD(nvjpegEncodeGetBufferSize);
    NJ_LOAD(nvjpegEncodeRetrieveBitstream);
    if (nj.nvjpegCreateSimple(&nj.jh) != NVJPEG_STATUS_SUCCESS) die("B nvjpegCreateSimple 失败");

    ck(cuStreamCreate(&g_bS, CU_STREAM_NON_BLOCKING), "B stream");
    if (nj.nvjpegEncoderStateCreate(nj.jh, &g_b_jstate, g_bS) != NVJPEG_STATUS_SUCCESS)
        die("B EncoderStateCreate 失败");
    if (nj.nvjpegEncoderParamsCreate(nj.jh, &g_b_jparams, g_bS) != NVJPEG_STATUS_SUCCESS)
        die("B EncoderParamsCreate 失败");
    nj.nvjpegEncoderParamsSetQuality(g_b_jparams, jpg_quality(), g_bS);
    nj.nvjpegEncoderParamsSetSamplingFactors(g_b_jparams, NVJPEG_CSS_420, g_bS);
    nj.nvjpegEncodeGetBufferSize(nj.jh, g_b_jparams, work_w, work_h, &g_b_jpg_cap);
    g_b_jpg = xmalloc(g_b_jpg_cap + 1024);
    ck(cuMemAlloc(&g_b_rgb_dev, (size_t)work_w * work_h * 3), "alloc B rgb");
    ck(cuMemAlloc(&g_b_rgb224_dev, BINSZ), "alloc B rgb224");
}

static void b_encode_one(int si) {
    CandidateSlot *c = &g_cand[si];
    unsigned long long a1 = (unsigned long long)c->dev;
    unsigned long long a2 = (unsigned long long)(c->dev + inUVoff);
    unsigned long long a6 = (unsigned long long)g_b_rgb_dev;
    int aP = (int)inPitch, aW = work_w, aH = work_h;
    void *kp[6] = { &a1, &aP, &a2, &aW, &aH, &a6 };
    ck(cuLaunchKernel(g_nv12_full, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1,
                      0, g_bS, kp, NULL), "B nv12_rgb_full");

    unsigned long long a224 = (unsigned long long)g_b_rgb224_dev;
    void *kp2[6] = { &a1, &aP, &a2, &aW, &aH, &a224 };
    ck(cuLaunchKernel(g_nv12_224, (SMALL_W * SMALL_H + 255) / 256, 1, 1, 256, 1, 1,
                      0, g_bS, kp2, NULL), "B nv12_rgb224");
    ck(cuMemcpyDtoHAsync(g_b_bin_buf, g_b_rgb224_dev, BINSZ, g_bS), "B dl rgb224");

    nvjpegImage_t src; memset(&src, 0, sizeof(src));
    src.channel[0] = (unsigned char *)g_b_rgb_dev;
    src.pitch[0] = (size_t)work_w * 3;
    size_t len = g_b_jpg_cap;
    if (nj.nvjpegEncodeImage(nj.jh, g_b_jstate, g_b_jparams, &src, NVJPEG_INPUT_RGBI,
                             work_w, work_h, g_bS) != NVJPEG_STATUS_SUCCESS)
        die("B nvjpegEncodeImage 失败");
    if (nj.nvjpegEncodeRetrieveBitstream(nj.jh, g_b_jstate, g_b_jpg, &len, g_bS) != NVJPEG_STATUS_SUCCESS)
        die("B nvjpegEncodeRetrieveBitstream 失败");
    ck(cuStreamSynchronize(g_bS), "B encode sync");

    mkdir_p(g_frames_dir);
    mkdir_p(g_dino_dir);
    char name[128];
    snprintf(name, sizeof(name), "%s_f%d.jpg", g_vhash, c->fn);
    write_frame_file(g_frames_dir, name, g_b_jpg, len);
    snprintf(name, sizeof(name), "%s_f%d.bin", g_vhash, c->fn);
    write_frame_file(g_dino_dir, name, g_b_bin_buf, BINSZ);
}

static void *extract_thread(void *arg) {
    (void)arg;
    ck(cuCtxPushCurrent(ctx), "push ctx B");
    mkdir_p(g_frames_dir);
    mkdir_p(g_dino_dir);
    b_init_encoder();
    int n_written = 0;
    for (;;) {
        int si = b_dequeue_slot();
        if (si < 0) break;
        b_encode_one(si);
        n_written++;
        b_release_slot(si);
    }
    printf("extract_thread: 写盘完成，%d 帧（jpg/bin）\n", n_written);
    fflush(stdout);
    if (g_b_rgb224_dev) { ck(cuMemFree(g_b_rgb224_dev), "free B rgb224"); g_b_rgb224_dev = 0; }
    if (g_b_rgb_dev) { ck(cuMemFree(g_b_rgb_dev), "free B rgb"); g_b_rgb_dev = 0; }
    if (g_b_jpg) { free(g_b_jpg); g_b_jpg = NULL; }
    if (g_b_jstate) { nj.nvjpegEncoderStateDestroy(g_b_jstate); g_b_jstate = NULL; }
    if (g_b_jparams) { nj.nvjpegEncoderParamsDestroy(g_b_jparams); g_b_jparams = NULL; }
    if (g_bS) { ck(cuStreamDestroy(g_bS), "destroy B stream"); g_bS = 0; }
    ck(cuCtxPopCurrent(NULL), "pop ctx B");
    return NULL;
}

static void b_start(void) {
    g_b_done = 0;
    if (pthread_create(&g_b_tid, NULL, extract_thread, NULL) != 0)
        die("pthread_create(B) 失败");
}

static void b_stop(void) {
    pthread_mutex_lock(&g_b_mu);
    g_b_done = 1;
    pthread_cond_broadcast(&g_b_has_work);
    pthread_mutex_unlock(&g_b_mu);
    pthread_join(g_b_tid, NULL);
}

/* ═══════════════════════════════════════════════════════
 * 在线状态机（逐位复刻 build_cuts/build_shots 语义）
 *  build_cuts:  过滤 c<=1||c>=total + MIN_GAP=2 相邻去重（丢弃当前，gap<2 不入列）
 *  build_shots: 二次过滤+qsort+完全去重 = 恒等（build_cuts 输出已有序唯一）
 *  → 在线确认 cut c_k 时 finalize shot (k-1)：[start=cf[k-2] 或 0, end=c_k-1]
 *    解码结束 finalize 最后 shot：[start=cf 末尾 或 0, end=total-1]
 *  c>=total 在线恒不触发（c=i-1 < i <= total，越界保护在 display_cb）
 * ═══════════════════════════════════════════════════════ */
static void finalize_shot(int id, int start, int end, int keep_from);

static void finalize_shot(int id, int start, int end, int keep_from) {
    (void)keep_from;
    if (id >= MAX_SHOTS) die("finalize_shot: Shot count exceeded MAX_SHOTS limit");
    if (end < start) die("finalize_shot: 空 shot（状态机错误）");
    Shot shot = {id, start, end};

    int desired_rep = 0, *desired_kfs = NULL, desired_n = 0;
    process_shot(&shot, &desired_rep, &desired_kfs, &desired_n);

    /* 把原算法选出的帧映射到本 shot 已缓存的真实 NV12 候选；
     * 大多数情况下 exact 命中，未命中才取最近低运动候选。 */
    int used[32] = {0};
    int *out = xmalloc(sizeof(int) * (desired_n ? desired_n : 1));
    int out_n = 0;
    int rep_out = -1;
    for (int i = 0; i < desired_n; i++) {
        int si = candidate_find_exact(desired_kfs[i]);
        if (si < 0) si = candidate_find_near(desired_kfs[i], 0);
        if (si < 0) continue;
        int ai = -1;
        for (int k = 0; k < g_active_n; k++) if (g_active_cand[k] == si) { ai = k; break; }
        if (ai >= 0 && used[ai]) continue;
        if (ai >= 0) used[ai] = 1;
        out[out_n++] = g_cand[si].fn;
        if (desired_kfs[i] == desired_rep) rep_out = g_cand[si].fn;
    }

    if (out_n == 0 && g_active_n > 0) {
        int best_i = 0;
        for (int i = 1; i < g_active_n; i++)
            if (g_cand[g_active_cand[i]].rank < g_cand[g_active_cand[best_i]].rank) best_i = i;
        int si = g_active_cand[best_i];
        out[out_n++] = g_cand[si].fn;
        rep_out = g_cand[si].fn;
        used[best_i] = 1;
    }
    if (rep_out < 0 && out_n > 0) rep_out = out[0];

    sk.shots[id] = shot;
    sk.n_shots = id + 1;
    online_reps[id] = rep_out >= 0 ? rep_out : desired_rep;
    online_nkfs[id] = out_n;
    online_kfs[id] = out;
    free(desired_kfs);

    /* 入队 B。非选中候选立即释放；选中候选 ownership 转给 B。 */
    for (int i = 0; i < g_active_n; i++) {
        int si = g_active_cand[i];
        if (used[i]) b_enqueue_slot(si);
        else b_release_slot(si);
    }
    g_active_n = 0;

    printf("  [shot %d] %d-%d -> rep=%d kfs=%d (B async)\n",
           id, start, end, online_reps[id], online_nkfs[id]);
    fflush(stdout);
}

/* 黑帧段独立 shot（2026-08-19 大名，判定照 ffmpeg blackdetect）：
 * blackdetect_eval 确认黑帧段 [seg_start, seg_end]（ratio>=pic_th 持续>=d）
 * 后调用。黑帧段前画面先收尾成普通 shot，黑帧段本身独立成 shot，段后开新
 * 普通 shot。防空：黑帧段不从已 finalize 区回退（scdet 切点已推进
 * current_shot_start 时，黑帧段从 current_shot_start 起，不重叠不空 shot）。 */

/* scdet 切点回调：在帧 c 发现切点 → 定稿当前 shot [current_shot_start, c-1]，推进起点 */
static void on_cut(int c) {
    if (c <= current_shot_start) return;         /* 防空 shot（cut 不早于当前起点） */
    finalize_shot(g_shot_id++, current_shot_start, c - 1, c);
    current_shot_start = c;
}

static void black_seg_split(int seg_start, int seg_end) {
    if (seg_end < seg_start) return;
    if (seg_start > current_shot_start) {          /* 黑帧段前画面收尾 */
        finalize_shot(g_shot_id++, current_shot_start, seg_start - 1, seg_start);
        current_shot_start = seg_start;
    }
    if (current_shot_start < seg_start) current_shot_start = seg_start;  /* 防空 */
    if (seg_end >= current_shot_start) {
        finalize_shot(g_shot_id++, current_shot_start, seg_end, seg_end + 1);
        current_shot_start = seg_end + 1;
    }
}

/* 队列（定义在下方；display_cb 先调用，需前向声明） */
static int q_take_empty(void);
static void q_put_full(int slot, int fn);

/* ═══════════════════════════════════════════════════════
 * display 回调（解码线程，薄版）：只做"map → 拷贝入队"（照 ffmpeg frame_thread 架构：
 * 解码线程产出帧入队，分析/选帧/保存全在 proc_thread；队列满则背压阻塞解码）
 * ═══════════════════════════════════════════════════════ */
/* ═══════════════════════════════════════════════════════
 * 帧队列（照 ffmpeg frame_thread 的 FQFrameQueue：mutex+cond 生产者-消费者）
 * 生产 = display_cb（解码线程），消费 = proc_thread（处理线程）；RING_CAP 满则背压
 * ═══════════════════════════════════════════════════════ */
static int q_take_empty(void) {   /* 解码线程：取空闲 slot（阻塞） */
    pthread_mutex_lock(&g_q_mu);
    while (g_empty_n == 0) pthread_cond_wait(&g_q_has_slot, &g_q_mu);
    int s = g_empty_stack[--g_empty_n];
    pthread_mutex_unlock(&g_q_mu);
    return s;
}
static void q_put_full(int slot, int fn) {
    pthread_mutex_lock(&g_q_mu);
    g_q_slot[(g_q_head + g_q_n) % RING_CAP] = slot;
    g_q_fn[(g_q_head + g_q_n) % RING_CAP] = fn;
    g_q_n++;
    pthread_cond_signal(&g_q_has_data);
    pthread_mutex_unlock(&g_q_mu);
}
static int q_take_full(int *slot, int *fn) {   /* 处理线程：取帧（阻塞；EOS 且空则返回 0） */
    pthread_mutex_lock(&g_q_mu);
    while (g_q_n == 0 && !g_eos) pthread_cond_wait(&g_q_has_data, &g_q_mu);
    if (g_q_n == 0 && g_eos) { pthread_mutex_unlock(&g_q_mu); return 0; }
    *slot = g_q_slot[g_q_head];
    *fn = g_q_fn[g_q_head];
    g_q_head = (g_q_head + 1) % RING_CAP;
    g_q_n--;
    pthread_mutex_unlock(&g_q_mu);
    return 1;
}
static void q_put_empty(int slot) {
    pthread_mutex_lock(&g_q_mu);
    g_empty_stack[g_empty_n++] = slot;
    pthread_cond_signal(&g_q_has_slot);
    pthread_mutex_unlock(&g_q_mu);
}

/* ═══════════════════════════════════════════════════════
 * 处理线程每帧：全链提交 procS → 1 次 sync → 状态机段（照抄原 display_cb 顺序）
 * scdet prev 占一槽（照 ffmpeg AVFrame 引用语义；freeze ref 已随裁剪移除），
 * shot 内每帧 jpg/bin 入内存缓冲，shot 结束时 finalize_shot 小循环写选中帧
 * ═══════════════════════════════════════════════════════ */
static int g_prev_shot_id = 0;

static void process_frame(int fn, int slot) {
    CUdeviceptr cur = dIn[slot];

    if (fn >= 1) {
        scdet_sad_kernel(dIn[g_prev_slot], cur, inPitch);
        ck(cuMemcpyDtoHAsync(&g_sad, g_sad_dev, 8, procS), "dl sad");
    } else {
        g_scores[0] = 0.0f;
        g_prev_mafd = 0.0;
    }
    blackdetect_count_kernel(cur, inPitch);
    ck(cuMemcpyDtoHAsync(&g_black, g_black_dev, 4, procS), "dl black");
    q_hist_kernel(cur, inPitch);
    q_tout_kernel(cur, inPitch);
    q_vrep_kernel(cur, inPitch);
    q_brng_kernel(cur, cur + inUVoff, inPitch);
    static uint32_t hist[256];
    static unsigned int bsum_h[NB_BLOCKS];
    static long long lap_h[NB_BLOCKS * 2];
    ck(cuMemcpyDtoHAsync(hist, g_q_hist_dev, 256 * 4, procS), "dl qhist");
    ck(cuMemcpyDtoHAsync(&g_q_tout_h, g_q_tout_dev, 4, procS), "dl qtout");
    ck(cuMemcpyDtoHAsync(&g_q_vrep_h, g_q_vrep_dev, 4, procS), "dl qvrep");
    ck(cuMemcpyDtoHAsync(&g_q_brng_h, g_q_brng_dev, 4, procS), "dl qbrng");

    unsigned long long y1 = (unsigned long long)cur;
    int yP = inPitch;
    void *kp_b[6] = { &y1, &yP, &work_w, &work_h, &g_small_dev, &g_bsum_dev };
    void *kp_l[2] = { &g_small_dev, &g_lap_dev };
    ck(cuLaunchKernel(g_ybox, NB_BLOCKS, 1, 1, 256, 1, 1, 0, procS, kp_b, NULL), "y224_box");
    ck(cuLaunchKernel(g_ylap, NB_BLOCKS, 1, 1, 256, 1, 1, 0, procS, kp_l, NULL), "y224_lap");
    ck(cuMemcpyDtoHAsync(bsum_h, g_bsum_dev, (size_t)NB_BLOCKS * 4, procS), "dl bsum");
    ck(cuMemcpyDtoHAsync(lap_h, g_lap_dev, (size_t)NB_BLOCKS * 16, procS), "dl lap");

    /* 当前 procS 还没同步：previous frame 的指标已经是 host-ready，
     * 此处提前提交 candidate D2D，随后一次 sync 同时等待分析和候选复制。 */
    if (fn >= 1 && g_prev_slot >= 0) {
        int prev_fn = fn - 1;
        candidate_consider(prev_fn, g_prev_slot, g_prev_shot_id);
    }

    ck(cuStreamSynchronize(procS), "sync proc");

    blackdetect_eval(fn);
    if (fn >= 1) {
        double mafd = (double)g_sad * 100.0 / ((double)work_w * work_h * 255.0);
        double diff = fabs(mafd - g_prev_mafd);
        double score = mafd < diff ? mafd : diff;
        if (score > 100) score = 100;
        if (score < 0) score = 0;
        g_prev_mafd = mafd;
        g_scores[fn] = (float)score;
        g_mafd[fn - 1] = (float)mafd;
        if (score >= SCDET_THR && g_n_cuts < MAX_CUTS) {
            g_cuts[g_n_cuts++] = fn - 1;
            on_cut(fn - 1);
        }
    }
    quality_eval(fn, hist);

    long long bs = 0, ls = 0, ls2 = 0;
    for (int b = 0; b < NB_BLOCKS; b++) {
        bs += (long long)bsum_h[b];
        ls += lap_h[b * 2];
        ls2 += lap_h[b * 2 + 1];
    }
    double cnt = 222.0 * 222.0;
    double mean = (double)ls / cnt;
    sharpness[fn] = (float)((double)ls2 / cnt - mean * mean);
    float bmean = (float)((double)bs / (SMALL_H * SMALL_W) / 255.0);
    float bv = 1.0f - fabsf(bmean - 0.5f) * 3.0f;
    if (bv < 0) bv = 0; if (bv > 1) bv = 1;
    brightness_arr[fn] = bv;

    /* current frame 归属于 cut 后的新 shot，供下一帧 candidate_consider 使用。 */
    g_prev_shot_id = g_shot_id;

    /* prev slot 现在已经被 candidate_consider 使用完；当前 slot 留作新的 prev。 */
    {
        int old_prev = g_prev_slot;
        g_prev_slot = slot;
        if (old_prev >= 0) q_put_empty(old_prev);
    }
}

/* 处理线程：消费队列 → process_frame → 队列空且 EOS → 收尾（照抄 decode 尾部） */
static void *proc_thread(void *arg) {
    (void)arg;
    ck(cuCtxPushCurrent(ctx), "push ctx proc");
    int slot = 0, fn = 0, g_last_fn = -1;
    while (q_take_full(&slot, &fn)) {
        process_frame(fn, slot);
        g_last_fn = fn;
    }

    int decoded = g_last_fn + 1;
    if (decoded < sk.total_frames) {
        fprintf(stderr, "unified_extract: 解码到 %d/%d 帧，total_frames 截断为 %d\n",
                decoded, sk.total_frames, decoded);
        sk.total_frames = decoded;
    }
    if (sk.total_frames < 2) die("解码帧数不足 2 帧，无法继续处理");

    /* 最后一帧没有下一帧可触发 candidate_consider；补一次。 */
    if (g_last_fn >= 0 && g_prev_slot >= 0) {
        candidate_consider(g_last_fn, g_prev_slot, g_prev_shot_id);
        ck(cuStreamSynchronize(procS), "final candidate sync");
    }

    if (black_start >= 0) {
        int last = decoded - 1; if (last < 0) last = 0;
        if ((last - black_start) >= (long long)(BLACK_DUR_SEC * sk.fps)) {
            if (black_n < MAX_BLACK) {
                black_seg_start[black_n] = black_start;
                black_seg_end[black_n] = last;
                black_n++;
            }
            black_seg_split(black_start, last);
        }
        black_start = -1;
    }

    if (current_shot_start < sk.total_frames)
        finalize_shot(g_shot_id++, current_shot_start, sk.total_frames - 1, -1);
    sk.n_shots = g_shot_id;

    if (g_prev_slot >= 0) {
        q_put_empty(g_prev_slot);
        g_prev_slot = -1;
    }

    cuCtxPopCurrent(NULL);
    return NULL;
}

/* ═══════════════════════════════════════════════════════
 * 写 JSON 前过滤 raw cuts（照抄 build_cuts：c<=1/c>=total skip + MIN_GAP 去重）
 * ═══════════════════════════════════════════════════════ */
static void build_cuts(void) {
    int n = 0;
    for (int i = 0; i < g_n_cuts; i++) {
        int c = g_cuts[i];
        if (c <= 1 || c >= sk.total_frames) continue;
        if (n == 0 || c - g_cuts[n - 1] >= MIN_GAP)
            g_cuts[n++] = c;
    }
    g_n_cuts = n;
}


/* ═══════════════════════════════════════════════════════
 * 单次解码会话（三段式接口）
 * ═══════════════════════════════════════════════════════ */
static void decoder_session_open(void) {
    /* 1. 创建 video parser（绑定 sequence/decode/display 回调） */
    CUVIDPARSERPARAMS pp; memset(&pp, 0, sizeof(pp));
    pp.CodecType = cudaVideoCodec_H264;
    pp.ulMaxNumDecodeSurfaces = 20;
    pp.ulMaxDisplayDelay = 4;
    pp.pUserData = NULL;
    pp.pfnSequenceCallback = sequence_cb;
    pp.pfnDecodePicture = decode_cb;
    pp.pfnDisplayPicture = display_cb;
    if (nv.cuvidCreateVideoParser(&g_parser, &pp) != CUDA_SUCCESS)
        die("cuvidCreateVideoParser 失败");

    /* 2. hako 上下文（avcC→annexb） */
    memset(&g_ab, 0, sizeof(g_ab));
    if (g_mov.v_extradata_size > 0 && annexb_open(&g_ab, g_mov.v_extradata, g_mov.v_extradata_size) < 0)
        die("annexb_open（avcC 解析）失败");

    /* 3. 样本缓冲 */
    g_sam = (uint8_t *)xmalloc(64 * 1024 * 1024);
    g_feed_failed = 0;

    /* 4. 启动处理线程（A 线：分析+选帧） */
    if (pthread_create(&g_proc_tid, NULL, proc_thread, NULL) != 0)
        die("pthread_create(A) 失败");
}

static void feed_video_once(void) {
    long n = 0;
    while (!g_feed_failed) {
        int r = mov_read_next(&g_mov, g_sam, &n);
        if (r == 0) break;            /* EOF */
        if (r < 0) { g_feed_failed = 1; break; }
        uint8_t *out = NULL; int osz = 0;
        if (annexb_filter(&g_ab, g_sam, (int)n, &out, &osz) < 0) { g_feed_failed = 1; break; }
        if (out) {
            CUVIDSOURCEDATAPACKET csp; memset(&csp, 0, sizeof(csp));
            csp.payload = out;
            csp.payload_size = (unsigned long)osz;
            if (nv.cuvidParseVideoData(g_parser, &csp) != CUDA_SUCCESS) {
                fprintf(stderr, "unified_extract: cuvidParseVideoData 失败\n");
                g_feed_failed = 1;
                free(out);
                break;
            }
            free(out);
        }
    }
    if (g_feed_failed) die("feed_video_once 失败");
}

static void decoder_session_close(void) {
    /* 1. EOS flush */
    CUVIDSOURCEDATAPACKET eos; memset(&eos, 0, sizeof(eos));
    eos.flags = CUVID_PKT_ENDOFSTREAM;
    nv.cuvidParseVideoData(g_parser, &eos);

    /* 2. 销毁 parser / decoder */
    if (g_parser)  { nv.cuvidDestroyVideoParser(g_parser);  g_parser  = NULL; }
    if (g_decoder) { nv.cuvidDestroyDecoder(g_decoder);     g_decoder = NULL; }

    /* 3. hako 收尾 */
    annexb_close(&g_ab);
    if (g_sam) { free(g_sam); g_sam = NULL; }

    /* 4. 通知并等待 A 线处理线程结束 */
    g_eos = 1;
    pthread_cond_broadcast(&g_q_has_data);
    pthread_join(g_proc_tid, NULL);
}

/* ═══════════════════════════════════════════════════════
 * 单遍流水：A 分析 + 少量候选缓存；B 异步编码/落盘
 * ═══════════════════════════════════════════════════════ */
static void decode_and_extract(const char *video) {
    if (sk.width >= sk.height) {
        work_h = 960;
        work_w = (int)((long)sk.width * 960 / sk.height);
    } else {
        work_w = 960;
        work_h = (int)((long)sk.height * 960 / sk.width);
    }
    if (work_w % 2) work_w++;
    if (work_h % 2) work_h++;
    printf("  工作分辨率: %dx%d（短边 960）\n", work_w, work_h);

    cuda_init(work_w, work_h);
    gpu_load_cubin("CUDA_KaKu.cubin");
    gpu_get_fn("scdet_sad", &g_fsad);
    gpu_get_fn("blackdetect_count", &g_fblack);
    gpu_get_fn("q_hist", &g_q_hist);
    gpu_get_fn("q_tout", &g_q_tout);
    gpu_get_fn("q_vrep", &g_q_vrep);
    gpu_get_fn("q_brng", &g_q_brng);
    gpu_get_fn("y224_box", &g_ybox);
    gpu_get_fn("y224_lap", &g_ylap);
    gpu_get_fn("nv12_rgb_full", &g_nv12_full);
    gpu_get_fn("nv12_rgb224", &g_nv12_224);
    ck(cuMemAlloc(&g_sad_dev, 8), "alloc sad");
    ck(cuMemAlloc(&g_black_dev, 4), "alloc black");
    ck(cuMemAlloc(&g_q_hist_dev, 256 * 4), "alloc q_hist");
    ck(cuMemAlloc(&g_q_tout_dev, 4), "alloc q_tout");
    ck(cuMemAlloc(&g_q_vrep_dev, 4), "alloc q_vrep");
    ck(cuMemAlloc(&g_q_brng_dev, 4), "alloc q_brng");
    ck(cuMemAlloc(&g_small_dev, GRAY_FSZ), "alloc small");
    ck(cuMemAlloc(&g_bsum_dev, (size_t)NB_BLOCKS * 4), "alloc bsum");
    ck(cuMemAlloc(&g_lap_dev, (size_t)NB_BLOCKS * 16), "alloc lap");

    g_nv12_bytes = (size_t)inUVoff + (size_t)inPitch * ((size_t)work_h / 2);
    candidate_pool_init();

    nvcuvid_open();

    /* B 线程独立 dlopen libnvjpeg 并建 encoder state/params（b_init_encoder）。A 线不持 nvjpeg。 */
    b_start();

    /* 单次解码：open → feed → close（A 线分析 + B 线异步抽帧全在一次解码内完成） */
    decoder_session_open();
    feed_video_once();
    decoder_session_close();

    /* A 已经把所有已定稿 shot 的候选交给 B，允许 B 收尾。 */
    b_stop();
}

/* ═══════════════════════════════════════════════════════
 * video_hash（方案 A，2026-08-18 大名定稿）：视频文件本身 SHA256 前 16 hex
 * ——业界标准密码学指纹（md5sum/openssl dgst 同源），与解码/GPU/每帧内容
 * 无关：文件不变哈希永远不变，彻底杜绝「每次跑不一样」
 * ═══════════════════════════════════════════════════════ */
static void compute_file_vhash(const char *path, char *out_hash, size_t out_n) {
    SHA256_CTX ctx; SHA256_Init(&ctx);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "unified_extract: 无法打开 %s 算文件哈希\n", path); exit(1); }
    static unsigned char buf[1 << 20];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), f)) > 0)
        SHA256_Update(&ctx, buf, r);
    fclose(f);
    unsigned char digest[32];
    SHA256_Final(digest, &ctx);
    for (int i = 0; i < 8; i++) snprintf(out_hash + i * 2, out_n - i * 2, "%02x", digest[i]);
    printf("  video_hash(文件 SHA256 前 16 hex): %.16s\n", out_hash);
}

static void write_raw_cuts(const char *path) {
    mkdir_for(path);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "unified_extract: 无法写 %s\n", path); exit(1); }
    fprintf(f, "{\n");
    fprintf(f, "  \"video\": \"%s\",\n", sk.video);
    fprintf(f, "  \"video_hash\": \"%s\",\n", sk.video_hash);
    fprintf(f, "  \"total_frames\": %d,\n", sk.total_frames);
    fprintf(f, "  \"duration\": %.6f,\n", g_mov.m_duration > 0 ? (double)g_mov.m_duration / g_mov.m_timescale : 0.0);
    fprintf(f, "  \"fps\": %.10g,\n", sk.fps);
    fprintf(f, "  \"fps_source\": \"avg_frame_rate\",\n");
    fprintf(f, "  \"width\": %d,\n", sk.width);
    fprintf(f, "  \"height\": %d,\n", sk.height);
    fprintf(f, "  \"cuts\": [\n");
    for (int i = 0; i < g_n_cuts; i++)
        fprintf(f, "    %d%s\n", g_cuts[i], i < g_n_cuts - 1 ? "," : "");
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("  -> %s (%d cuts)\n", path, g_n_cuts);
}

static void write_skeleton(const char *path, const char *vid_name, const char *project_id) {
    mkdir_for(path);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "unified_extract: 无法写 %s\n", path); exit(1); }
    fprintf(f, "{\n");
    fprintf(f, "  \"video\": \"%s\",\n", sk.video);
    fprintf(f, "  \"video_hash\": \"%s\",\n", sk.video_hash);
    fprintf(f, "  \"video_id\": \"%s\",\n", vid_name);
    fprintf(f, "  \"project_id\": \"%s\",\n", project_id);
    fprintf(f, "  \"fps\": %.10g,\n", sk.fps);
    fprintf(f, "  \"width\": %d,\n", sk.width);
    fprintf(f, "  \"height\": %d,\n", sk.height);
    fprintf(f, "  \"total_frames\": %d,\n", sk.total_frames);
    fprintf(f, "  \"shots\": [\n");
    for (int i = 0; i < sk.n_shots; i++) {
        fprintf(f, "    {\n");
        fprintf(f, "      \"id\": %d,\n", i);
        fprintf(f, "      \"range\": {\n");
        fprintf(f, "        \"start\": %d,\n", sk.shots[i].start);
        fprintf(f, "        \"end\": %d\n", sk.shots[i].end);
        fprintf(f, "      }\n");
        fprintf(f, "    }%s\n", (i < sk.n_shots - 1) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("  -> %s (%d shots)\n", path, sk.n_shots);
}

static void write_features(const char *path) {
    mkdir_for(path);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "unified_extract: 无法写 %s\n", path); exit(1); }
    fprintf(f, "{\n");
    fprintf(f, "  \"video\": \"%s\",\n", sk.video);
    fprintf(f, "  \"fps\": %.10g,\n", sk.fps);
    fprintf(f, "  \"total_frames\": %d,\n", sk.total_frames);
    fprintf(f, "  \"scd_scores\": [");
    for (int i = 0; i < sk.total_frames; i++)
        fprintf(f, "%s%.3f", i ? "," : "", g_scores[i]);
    fprintf(f, "],\n");
    fprintf(f, "  \"mafd\": [");
    for (int i = 0; i < sk.total_frames - 1; i++)
        fprintf(f, "%s%.5f", i ? "," : "", g_mafd[i]);
    fprintf(f, "],\n");
    fprintf(f, "  \"quality\": [\n");
    for (int i = 0; i < sk.total_frames; i++) {
        float *q = &g_quality[(size_t)i * 9];
        if (i) fprintf(f, ",\n");
        fprintf(f, "    {\"yavg\": %.5g, \"ylow\": %.0f, \"yhigh\": %.0f, \"ymin\": %.0f, \"ymax\": %.0f, "
                   "\"tout\": %.5g, \"vrep\": %.5g, \"brng\": %.5g, \"entropy\": %.5g}",
                q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]);
    }
    fprintf(f, "\n  ],\n");
    fprintf(f, "  \"sharpness\": [");
    for (int i = 0; i < sk.total_frames; i++)
        fprintf(f, "%s%.5f", i ? "," : "", sharpness[i]);
    fprintf(f, "]\n}\n");
    fclose(f);
    printf("  -> %s\n", path);
}

static void write_events(const char *path) {
    mkdir_for(path);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "  \"video\": \"%s\",\n", sk.video);
    fprintf(f, "  \"video_hash\": \"%s\",\n", sk.video_hash);
    fprintf(f, "  \"black_segments\": [\n");
    for (int i = 0; i < black_n; i++) {
        if (i) fprintf(f, ",\n");
        fprintf(f, "    {\"type\": \"black\", \"start_frame\": %d, \"end_frame\": %d, \"duration\": %.3f}",
                black_seg_start[i], black_seg_end[i],
                (double)(black_seg_end[i] - black_seg_start[i]) / sk.fps);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("  -> %s\n", path);
}

static void write_output(const char *out_path, int *reps, int **kfs, int *nkfs) {
    mkdir_for(out_path);
    FILE *f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "unified_extract: 无法写 %s\n", out_path); exit(1); }
    fprintf(f, "{\n");
    fprintf(f, "  \"video\": \"%s\",\n", sk.video);
    fprintf(f, "  \"video_hash\": \"%s\",\n", sk.video_hash);
    fprintf(f, "  \"fps\": %.10g,\n", sk.fps);
    fprintf(f, "  \"width\": %d,\n", sk.width);
    fprintf(f, "  \"height\": %d,\n", sk.height);
    fprintf(f, "  \"total_frames\": %d,\n", sk.total_frames);
    fprintf(f, "  \"shots\": [\n");
    for (int i = 0; i < sk.n_shots; i++) {
        fprintf(f, "    {\"id\": %d, \"range\": {\"start\": %d, \"end\": %d},\n",
                sk.shots[i].id, sk.shots[i].start, sk.shots[i].end);
        fprintf(f, "     \"representative_frame\": %d,\n", reps[i]);
        fprintf(f, "     \"key_frames\": [");
        for (int k = 0; k < nkfs[i]; k++) {
            fprintf(f, "%d%s", kfs[i][k], k < nkfs[i] - 1 ? ", " : "");
        }
        fprintf(f, "]%s\n", i < sk.n_shots - 1 ? "}," : "}");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);

    int total_kf = 0;
    for (int i = 0; i < sk.n_shots; i++) total_kf += nkfs[i];
    printf("Done: %d shots, %d total key_frames -> %s\n", sk.n_shots, total_kf, out_path);
}

/* ═══════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    const char *mode_b_project = NULL, *mode_b_vid = NULL;
    const char *out_root_arg = NULL;
    const char *pos[4] = {0};
    int npos = 0;
    for (int i = 1; i < argc && npos < 4; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) { out_root_arg = argv[++i]; }
        else pos[npos++] = argv[i];
    }
    if (npos == 1) {
    } else if (npos == 3) {
        mode_b_project = pos[1];
        mode_b_vid = pos[2];
    } else {
        fprintf(stderr, "用法: %s <视频> [<project> <vid_name>] [-o <输出根>]\n", argv[0]);
        return 1;
    }
    const char *video = pos[0];

    get_script_dir();

    /* S2 元数据 C 内化：自写 mov box 解析（抄 mov.c）替代 libavformat */
    if (mov_open(&g_mov, video) < 0) {
        fprintf(stderr, "unified_extract: mov_open 失败: %s\n", video);
        return 1;
    }
    if (!g_mov.has_video) { fprintf(stderr, "unified_extract: 无视频轨\n"); return 1; }
    snprintf(sk.video, sizeof(sk.video), "%s", video);
    sk.width = (int)g_mov.width;
    sk.height = (int)g_mov.height;
    sk.total_frames = (int)g_mov.v_nb_frames;
    sk.fps = g_mov.v_fps_den > 0 ? (double)g_mov.v_fps_num / g_mov.v_fps_den : 0.0;
    if (sk.total_frames <= 0 && g_mov.m_duration > 0 && sk.fps > 0)
        sk.total_frames = (int)llround(g_mov.m_duration * sk.fps);
    if (sk.width <= 0 || sk.height <= 0 || sk.fps <= 0 || sk.total_frames <= 0) {
        fprintf(stderr, "unified_extract: 元数据读取失败\n");
        return 1;
    }
    printf("[unified_extract] %dx%d  %dfr  %.1fs  %.4ffps\n",
           sk.width, sk.height, sk.total_frames,
           g_mov.m_duration > 0 ? (double)g_mov.m_duration / g_mov.m_timescale : 0.0, sk.fps);
    fflush(stdout);
    if (sk.total_frames < 2) die("total_frames < 2");

    /* vid_name / out_root（照抄 00 Mode A/B） */
    const char *base = strrchr(video, '/'); base = base ? base + 1 : video;
    char vid_name[512];
    const char *dot = strrchr(base, '.');
    if (dot) { size_t nl = dot - base; memcpy(vid_name, base, nl); vid_name[nl] = 0; }
    else snprintf(vid_name, sizeof(vid_name), "%s", base);
    if (mode_b_project) snprintf(vid_name, sizeof(vid_name), "%s", mode_b_vid);

    /* video_hash = 文件 SHA256（方案 A）：解码前算好。产物文件名前缀取前 6 位（内容指纹，
     * 2026-08-19 大名定稿；fnv6 换名实效哈希已禁——g_vhash 不再由视频文件名算） */
    compute_file_vhash(video, sk.video_hash, sizeof(sk.video_hash));
    sk.video_hash[6] = '\0';                       /* JSON 字段 video_hash 取前 6 位（大名定稿） */
    snprintf(g_vhash, sizeof(g_vhash), "%s", sk.video_hash);
    /* 默认输出根 = 视频源目录（领导 2026-08-07：纯工具不设 -o 时默认输出在视频源目录） */
    char def_out[4096];
    const char *slash = strrchr(video, '/');
    if (slash) {
        size_t nl = (size_t)(slash - video);
        memcpy(def_out, video, nl);
        def_out[nl] = 0;
    } else {
        snprintf(def_out, sizeof(def_out), ".");
    }
    const char *out_root = out_root_arg ? out_root_arg
                          : (mode_b_project ? mode_b_project : def_out);

    /* 输出路径（全产出） */
    snprintf(g_out_raw,  sizeof(g_out_raw),  "%s/shikomi/cuts/%s_cuts.json",
             out_root, g_vhash);
    snprintf(g_out_skel, sizeof(g_out_skel), "%s/shikomi/skeleton/%s_skeleton.json",
             out_root, g_vhash);
    snprintf(g_out_sf,   sizeof(g_out_sf),   "%s/shikomi/select_frames/%s_select_frames.json",
             out_root, g_vhash);
    snprintf(g_out_events, sizeof(g_out_events), "%s/shikomi/events/%s_events.json",
             out_root, g_vhash);
    snprintf(g_out_feat, sizeof(g_out_feat), "%s/shikomi/features/%s_features.json",
             out_root, g_vhash);
    snprintf(g_frames_dir, sizeof(g_frames_dir), "%s/shikomi/frames", out_root);
    snprintf(g_dino_dir,   sizeof(g_dino_dir),   "%s/shikomi/frames224", out_root);

    /* 数组（全标量：帧曲线/特征/指纹，像素零驻留零落盘） */
    g_mafd = xmalloc(sizeof(float) * (sk.total_frames - 1));
    g_scores = xmalloc(sizeof(float) * sk.total_frames);
    g_quality = xmalloc(sizeof(float) * sk.total_frames * 9);
    sharpness = xmalloc(sizeof(float) * sk.total_frames);
    brightness_arr = xmalloc(sizeof(float) * sk.total_frames);

    /* 在线选帧输出槽（shots 数 = cuts+1+黑帧段数 <= MAX_SHOTS） */
    sk.shots = xmalloc(sizeof(Shot) * MAX_SHOTS);
    online_reps = xmalloc(sizeof(int) * MAX_SHOTS);
    online_kfs = xmalloc(sizeof(int *) * MAX_SHOTS);
    online_nkfs = xmalloc(sizeof(int) * MAX_SHOTS);
    sk.n_shots = 0;

    /* video_hash（文件 SHA256 前 6 位）已在输出路径设置前算好。 */

    printf("Phase 1/1: decode + analysis + async extraction...\n");
    decode_and_extract(video);

    /* JSON 产物；jpg/bin 已由 B 在线异步生成。 */
    build_cuts();              /* 写 JSON 前过滤 raw cuts（照抄 Preproc：c<=1/c>=total skip + MIN_GAP） */
    write_raw_cuts(g_out_raw);
    write_skeleton(g_out_skel, vid_name, mode_b_project ? mode_b_project : vid_name);
    write_features(g_out_feat);
    write_events(g_out_events);
    write_output(g_out_sf, online_reps, online_kfs, online_nkfs);

    if (nj.jh) nj.nvjpegDestroy(nj.jh);
    if (nj.lib) dlclose(nj.lib);
    cuda_destroy();

    free(g_mafd); free(g_scores);
    free(g_quality); free(sharpness);
    free(brightness_arr);
    for (int i = 0; i < sk.n_shots; i++) free(online_kfs[i]);
    free(online_kfs); free(online_nkfs); free(online_reps);
    free(sk.shots);
    mov_close(&g_mov);
    return 0;
}