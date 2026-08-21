/*
 * preproc/duo_analyze.c — 双线并发·线 A（分析 + 选帧 + 共享产物；2026-08-18 大名定稿）
 *
 * 用法: ./duo_analyze <视频> [<project> <vid_name>] [-o <输出根>]
 * 依赖: preproc/unified_kernels.cubin（nvcc 编；切点源一字不动 = 内部 CUDA scdet_sad）、
 *       自写 mov 解析 mp4_mov.c、NVCUVID + nvjpeg；video_hash = 文件 SHA256（方案 A）
 * 产物: preproc/cuts|skeleton|select_frames|events|features/<hash>_*.json + 帧文件（jpg/bin，线 B 写盘）
 *
 * 背景：unified_extract 单遍把「分析 + 全分辨率抽帧」串在一起，每帧无条件 nvjpeg 编码
 * （163822 帧全编码只写 1530 张）拖慢整条。双线架构（同进程双线程绑定，2026-08-19
 * 大名定稿：B 本来就跟 A 一起启动，一个 C 程序内搞定，无需跨进程 IPC）：
 *   线 A（主线程，唯一解码 session）：解码 → 切点 + 全分析链 + 在线选帧，shot 定稿时
 *       选中帧 jpg/bin → 共享显存字节环（g_share_dev），定稿消息 → 内存环形队列；A 全速跑，零等待 B
 *   线 B（extract_thread，写盘线程）：读消息 → 环上按 off/len 分片 D2H → 写盘即用即丢
 *       （照抄现役 write_frame_file）；同进程同 CUDA 上下文，无 IPC 无等待握手
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
 *   nvcc -arch=all -fatbin -o unified_kernels.cubin unified_kernels.cu
 *   /usr/bin/gcc -O2 -o duo_analyze duo_analyze.c mp4_mov.c \
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
#include "mp4_mov.h"                     /* 自写 MOV 解析 + avcC→annexb */

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

/* ── 双线共享产物（线 A：shot NV12 池 + 共享显存环 + 定稿消息队列）──
 * 2026-08-18 大名定稿：A 全速跑不等 B；shot 定稿时选中帧 NV12 从 shot 池拷入
 * 共享显存环（CUDA IPC），定稿消息经共享内存队列发出；B 无解码器，读消息 →
 * 共享显存拿 NV12 → nvjpeg 编码写盘 */
#define BINSZ           (224 * 224 * 3)   /* 224×224×3 = 150528（DINO 小图，照抄现役） */
#define SHOT_BUF_MAX    (16LL * 1024 * 1024 * 1024)  /* shot 缓冲防御上限（jpg 流；单镜头超长时停） */
#define SHARE_MB        64          /* 共享显存环容量 MB（UE_SHARE_MB 可覆盖） */
#define SHM_MSG_CAP        256         /* 定稿消息条数（内存环形队列，B 线程消费） */
#define SHM_MSG_LEN        512         /* 单条消息长度（shot id/start/end/槽位/帧号列表） */

typedef struct {
    uint32_t overflow;                 /* B 消费不及（环/消息溢出，产物作废） */
    uint32_t head, tail;               /* 共享显存环 SPSC 字节指针（A 写 tail / B 读 head，循环） */
    uint32_t msg_head, msg_tail, msg_n;/* 定稿消息环形队列（A 写/B 读） */
    char     msg[SHM_MSG_CAP][SHM_MSG_LEN];
    uint64_t ring_bytes;               /* 环容量字节（环形寻址用） */
} ShareHdr;
static ShareHdr g_share;               /* 进程内共享（A 主线程写/B 线程读，同 CUDA 上下文） */

/* B 线程唤醒（A 发消息 / g_proc_done 置位时 broadcast；A 永不等 B） */
static pthread_mutex_t g_b_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_b_cond = PTHREAD_COND_INITIALIZER;
static pthread_t       g_b_tid;        /* 写盘线程（B） */

/* ── 编码链（照抄现役 unified_extract）：nvjpeg jpg + 224 bin → host shot 缓冲 ── */
static CUfunction g_nv12_full = 0;     /* nv12_rgb_full */
static CUfunction g_nv12_224 = 0;      /* nv12_rgb224 */
static CUdeviceptr g_rgb_dev = 0;      /* work 尺寸 w*h*3（nvjpeg 编码输入，零 DtoH） */
static CUdeviceptr g_rgb224_dev = 0;   /* 224×224×3（bin） */
static unsigned char *g_jpg = NULL;    /* 压缩流 host 缓冲（GetBufferSize 定容一次） */
static size_t g_jpg_cap = 0;
static unsigned char g_bin_buf[BINSZ]; /* 224 bin host 中转（每帧 DTOH 后入 shot 缓冲） */

/* nvjpeg dlopen 句柄 + 函数指针（照抄现役） */
typedef struct {
    void *lib;
    nvjpegHandle_t             jh;
    nvjpegEncoderState_t       jstate;
    nvjpegEncoderParams_t      jparams;
    nvjpegStatus_t (*nvjpegCreateSimple)(nvjpegHandle_t *);
    nvjpegStatus_t (*nvjpegDestroy)(nvjpegHandle_t);
    nvjpegStatus_t (*nvjpegEncoderStateCreate)(nvjpegHandle_t, nvjpegEncoderState_t *, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncoderStateDestroy)(nvjpegEncoderState_t);
    nvjpegStatus_t (*nvjpegEncoderParamsCreate)(nvjpegHandle_t, nvjpegEncoderParams_t *, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncoderParamsDestroy)(nvjpegEncoderParams_t);
    nvjpegStatus_t (*nvjpegEncoderParamsSetQuality)(nvjpegEncoderParams_t, int, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncoderParamsSetSamplingFactors)(nvjpegEncoderParams_t, nvjpegChromaSubsampling_t, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncodeImage)(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t,
                                        const nvjpegImage_t *, nvjpegInputFormat_t, int, int, cudaStream_t);
    nvjpegStatus_t (*nvjpegEncodeGetBufferSize)(nvjpegHandle_t, const nvjpegEncoderParams_t, int, int, size_t *);
    nvjpegStatus_t (*nvjpegEncodeRetrieveBitstream)(nvjpegHandle_t, nvjpegEncoderState_t,
                                                    unsigned char *, size_t *, cudaStream_t);
} NVJPEG;
static NVJPEG nj;

#define NJ_LOAD(name) do { *(void **)&nj.name = dlsym(nj.lib, #name); \
    if (!nj.name) { fprintf(stderr, "dlsym " #name " 失败\n"); exit(1); } } while (0)

/* shot 级内存缓冲（照抄现役；帧出解码器即编码入缓冲，shot 定稿取选中帧共享给 B） */
typedef struct {
    unsigned char *jpg;                /* jpg 连续流 */
    size_t *off, *len;                 /* 每帧 jpg 偏移/长度 */
    unsigned char *bins;               /* 每帧 224×224×3 连续区 */
    int n, cap;                        /* 已 push 帧数/容量 */
    size_t jpg_used, jpg_cap;          /* jpg 流使用量/容量 */
    int start;                         /* shot 起始帧号（open 时记录，finalize 断言用） */
    int open;
} ShotBuf;
static ShotBuf g_sb;

static CUdeviceptr g_share_dev = 0;    /* 共享显存环（UE_SHARE_MB 字节流，进程内） */
static int      g_share_mb = SHARE_MB; /* 环容量 MB（UE_SHARE_MB 覆盖） */
static int      g_share_on = 0;        /* 共享区使能（UE_NO_SHARE=1 关闭） */
static char     g_frames_dir[4096], g_dino_dir[4096];  /* B 线程写盘目录（main 算好） */

/* ── 在线选帧（逐位复刻 build_cuts/build_shots 语义）────── */
static int g_cf[MAX_CUTS];             /* 已确认 cuts（build_cuts 过滤后列表） */
static int g_n_cf = 0;
static int *online_reps = NULL;        /* [MAX_SHOTS] 每 shot 代表帧 */
static int **online_kfs = NULL;        /* [MAX_SHOTS] 每 shot 关键帧 */
static int *online_nkfs = NULL;        /* [MAX_SHOTS] 每 shot 关键帧数 */

/* ── shot 状态机（2026-08-19 大名：黑帧段 = 独立 shot）──
 * 切点唯一 = scdet（score>=threshold，照抄 vf_scdet_vulkan.c，无黑帧干预）；
 * 黑帧段独立 shot 判定照抄 vf_blackdetect_vulkan.c（ratio>=pic_th=0.98 持续
 * >=d=2.0s），blackdetect_eval 确认后 black_seg_split 拆段（定义见
 * finalize_shot 之后；先声明供 blackdetect_eval 调用）。 */
static int  g_shot_id = 0;             /* shot 编号（scdet cut + 黑帧段切点共用） */
static int  current_shot_start = 0;    /* 当前 open shot 起始帧 */
static void black_seg_split(int seg_start, int seg_end);

/* 编码链说明：抽帧路径照抄现役 unified_extract（nv12_rgb_full → nvjpeg jpg +
 * nv12_rgb224 → bin），帧出解码器即编码入 g_sb 内存缓冲（~250KB/帧 vs 旧
 * NV12 池 2.1MB/帧，8 倍省），shot 定稿取选中帧共享给 B（B GPU 写盘） */

/* ── 输出路径（main 算好） ─────────────────────────────── */
static char g_out_raw[4096], g_out_skel[4096], g_out_sf[4096];
static char g_out_events[4096], g_out_feat[4096];
static char g_vhash[32] = "";          /* 产物文件名前缀 = video_hash 前 6 位（内容指纹，2026-08-19 大名定稿；fnv6 换名实效哈希已禁） */
static MOV g_mov;                      /* S2/S3: 自写 mov 容器（抄 ffmpeg 9.0 mov.c） */

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

/* nvjpeg dlopen（系统无 libnvjpeg → conda env 多路径兜底，照抄现役） */
static void nvjpeg_open(void) {
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
}

/* ═══════════════════════════════════════════════════════
 * shot 缓冲（内存；帧出解码器即编码入缓冲，shot 定稿取选中帧共享给 B）
 * ═══════════════════════════════════════════════════════ */
static void sb_init(void) {
    memset(&g_sb, 0, sizeof(g_sb));
    g_sb.cap = 256;
    g_sb.off = xmalloc(sizeof(size_t) * g_sb.cap);
    g_sb.len = xmalloc(sizeof(size_t) * g_sb.cap);
    g_sb.bins = xmalloc((size_t)g_sb.cap * BINSZ);
    g_sb.jpg_cap = 1 << 20;   /* 1MB 起，翻倍 */
    g_sb.jpg = xmalloc(g_sb.jpg_cap);
}

static void sb_push(int fn, const unsigned char *jpg, size_t jpg_len, const unsigned char *bin) {
    if (!g_sb.open) { g_sb.open = 1; g_sb.start = fn; g_sb.n = 0; g_sb.jpg_used = 0; }
    if (g_sb.n == g_sb.cap) {
        int nc = g_sb.cap * 2;
        g_sb.off = realloc(g_sb.off, sizeof(size_t) * nc);
        g_sb.len = realloc(g_sb.len, sizeof(size_t) * nc);
        g_sb.bins = realloc(g_sb.bins, (size_t)nc * BINSZ);
        if (!g_sb.off || !g_sb.len || !g_sb.bins) die("shot 缓冲 realloc 失败");
        g_sb.cap = nc;
    }
    if (g_sb.jpg_used + jpg_len > g_sb.jpg_cap) {
        size_t nc = g_sb.jpg_cap;
        while (nc < g_sb.jpg_used + jpg_len) nc *= 2;
        if (nc > (size_t)SHOT_BUF_MAX) die("shot 缓冲超过上限（超长单镜头？）");
        g_sb.jpg = realloc(g_sb.jpg, nc);
        if (!g_sb.jpg) die("shot jpg realloc 失败");
        g_sb.jpg_cap = nc;
    }
    g_sb.off[g_sb.n] = g_sb.jpg_used;
    g_sb.len[g_sb.n] = jpg_len;
    memcpy(g_sb.jpg + g_sb.jpg_used, jpg, jpg_len);
    g_sb.jpg_used += jpg_len;
    memcpy(g_sb.bins + (size_t)g_sb.n * BINSZ, bin, BINSZ);
    g_sb.n++;
    if (g_sb.jpg_used > (size_t)SHOT_BUF_MAX) die("shot 缓冲超过上限（超长单镜头？）");
}

static void sb_free(void) {
    if (g_sb.jpg)  { free(g_sb.jpg);  g_sb.jpg  = NULL; }
    if (g_sb.off)  { free(g_sb.off);  g_sb.off  = NULL; }
    if (g_sb.len)  { free(g_sb.len);  g_sb.len  = NULL; }
    if (g_sb.bins) { free(g_sb.bins); g_sb.bins = NULL; }
    memset(&g_sb, 0, sizeof(g_sb));
}

static void mkdir_p(const char *path) {
    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    mkdir(tmp, 0755);
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
static CUstream        inS = 0;         /* 解码线程：帧拷贝（display_cb 内，legacy） */
static CUstream        procS = 0;       /* 处理线程：分析链 + 编码（NON_BLOCKING，与 inS 并行） */
#define RING_CAP        64              /* 帧环形缓冲（背压上限；1080p NV12 ≈ 200MB） */
static CUdeviceptr     dIn[RING_CAP];   /* 帧环形缓冲（display 拷贝入队，处理线程消费） */
static CUevent         g_in_ready[RING_CAP]; /* 每槽：inS 流 D2D 完成事件（procS 跨流等待） */
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
    ck(cuStreamCreate(&inS, CU_STREAM_NON_BLOCKING), "inStream");
    ck(cuStreamCreate(&procS, CU_STREAM_NON_BLOCKING), "procStream");

    /* 帧环形缓冲（NV12 布局：Y 平面 + UV 交错） */
    inPitch  = (uint32_t)(((size_t)w + 127) & ~(size_t)127);   /* 2D 拷贝行对齐 */
    inUVoff  = inPitch * (uint32_t)h;
    size_t nv12 = (size_t)inUVoff + (size_t)inPitch * ((uint32_t)h / 2);
    for (int i = 0; i < RING_CAP; i++) {
        char nm[32]; snprintf(nm, sizeof(nm), "alloc dIn%d", i);
        ck(cuMemAlloc(&dIn[i], nv12), nm);
        ck(cuEventCreate(&g_in_ready[i], CU_EVENT_DISABLE_TIMING), "event");
        g_empty_stack[g_empty_n++] = i;
    }
}

/* 上传一帧 NV12（Y 平面 + UV 交错平面）——显存源（NVCUVID 解码帧）GPU→GPU，零 CPU 往返 */
static void upload_nv12_dev(CUdeviceptr src, size_t srcPitch, CUdeviceptr dst, CUstream s) {
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
    if (g_sad_dev) ck(cuMemFree(g_sad_dev), "free sad");
    if (g_black_dev) ck(cuMemFree(g_black_dev), "free black");
    if (g_q_brng_dev) ck(cuMemFree(g_q_brng_dev), "free q_brng");
    if (g_q_vrep_dev) ck(cuMemFree(g_q_vrep_dev), "free q_vrep");
    if (g_q_tout_dev) ck(cuMemFree(g_q_tout_dev), "free q_tout");
    if (g_q_hist_dev) ck(cuMemFree(g_q_hist_dev), "free q_hist");
    if (g_lap_dev) ck(cuMemFree(g_lap_dev), "free lap");
    if (g_bsum_dev) ck(cuMemFree(g_bsum_dev), "free bsum");
    if (g_small_dev) ck(cuMemFree(g_small_dev), "free small");
    if (g_rgb224_dev) ck(cuMemFree(g_rgb224_dev), "free rgb224");
    if (g_rgb_dev) ck(cuMemFree(g_rgb_dev), "free rgb");
    for (int i = 0; i < RING_CAP; i++)
        if (dIn[i]) ck(cuMemFree(dIn[i]), "free dIn-ring");
    if (procS) ck(cuStreamDestroy(procS), "destroy procS");
    if (inS) ck(cuStreamDestroy(inS), "destroy inS");
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

/* ═══════════════════════════════════════════════════════
 * 共享区（A 侧；B 无解码器，读消息 → 共享显存拿 NV12 → nvjpeg 写盘）
 * ═══════════════════════════════════════════════════════ */
/* 共享显存环初始化（进程内：A 主线程写 / B 写盘线程读，同 CUDA 上下文） */
static void share_init(void) {
    if (getenv("UE_NO_SHARE")) {
        printf("  [share] UE_NO_SHARE=1，跳过共享区（纯分析单跑）\n");
        fflush(stdout);
        return;
    }
    const char *e;
    if ((e = getenv("UE_SHARE_MB"))) {
        int v = atoi(e);
        if (v >= 4 && v <= 1024) g_share_mb = v;
    }

    /* SPSC 字节环：A 写 tail / B 读 head；B 追着消费即用即丢，循环不炸
     * （B 写盘跟不上才 overflow → 整条重跑）。 */
    ck(cuMemAlloc(&g_share_dev, (size_t)g_share_mb * 1024 * 1024), "alloc share ring");
    memset(&g_share, 0, sizeof(g_share));
    g_share.ring_bytes = (uint64_t)g_share_mb * 1024 * 1024;
    g_share_on = 1;
    printf("  [share] 共享环: %d MB（字节流，SPSC；B 线程绑定写盘）\n", g_share_mb);
    fflush(stdout);
}

/* 环写（分片处理跨边界；与 B 的读对称，按 off/len 还原块） */
static void ring_write(uint64_t pos, const void *src, size_t len) {
    uint64_t cap = g_share.ring_bytes;
    size_t done = 0;
    while (done < len) {
        size_t off = (size_t)(pos % cap);
        size_t chunk = len - done;
        if (off + chunk > cap) chunk = cap - off;
        ck(cuMemcpyHtoDAsync(g_share_dev + off, (const unsigned char *)src + done, chunk, procS),
           "ring write");
        pos += chunk; done += chunk;
    }
}

/* ═══════════════════════════════════════════════════════
 * 线 B：写盘线程（同进程绑定；读消息 → 环上分片 D2H → 写盘即用即丢）
 * ═══════════════════════════════════════════════════════ */
/* 环形 D2H（分片处理跨边界；同步拷贝 = pageable 安全路径，照抄
 * FFmpeg hwcontext_cuda.c cuda_transfer_data 的 cuMemcpy2DAsync+sync 语义） */
static void ring_read(uint64_t pos, void *dst, size_t len) {
    uint64_t cap = g_share.ring_bytes;
    size_t done = 0;
    while (done < len) {
        size_t off = (size_t)((pos + done) % cap);
        size_t chunk = len - done;
        if (off + chunk > cap) chunk = cap - off;
        ck(cuMemcpyDtoH((unsigned char *)dst + done, g_share_dev + off, chunk), "ring read");
        done += chunk;
    }
}

/* 原子写（.tmp + rename，幂等；已存在跳过——照抄现役 write_frame_file） */
static void write_frame_file(const char *dir, const char *name, const void *data, size_t len) {
    char tmp[8192], fin[8192];
    snprintf(fin, sizeof(fin), "%s/%s", dir, name);
    if (access(fin, F_OK) == 0) return;
    snprintf(tmp, sizeof(tmp), "%s/.%s.tmp", dir, name);
    FILE *f = fopen(tmp, "wb");
    if (f) {
        if (fwrite(data, 1, len, f) == len) {
            fclose(f);
            if (rename(tmp, fin) != 0) fprintf(stderr, "rename %s 失败\n", tmp);
        } else { fclose(f); fprintf(stderr, "fwrite %s 失败\n", tmp); }
    } else fprintf(stderr, "fopen %s 失败\n", tmp);
}

/* B 写盘线程：消费消息 → 按 off/len 分片 D2H → 写盘即用即丢（不攒缓冲）。
 * A 永不等 B：数据写完 sync 才推 tail、发消息，B 拿到消息时数据必已就绪；
 * B 消费完推进 head，A 才不误判环满。g_proc_done 置位（proc 尾消息全入队）且消息
 * 消费完 → 退出。不能用 g_eos：g_eos 在 proc 尾部 finalize 之前就置位，B 若在
 * 处理线程入队末 shot 消息前醒来会提前退丢末 shot（EP01 实测丢 163767）。 */
static void *extract_thread(void *arg) {
    (void)arg;
    ck(cuCtxPushCurrent(ctx), "push ctx extract");
    mkdir_p(g_frames_dir);
    mkdir_p(g_dino_dir);
    unsigned char *jpg_buf = xmalloc(1 << 20);
    size_t jpg_cap = 1 << 20;
    unsigned char *bin_buf = xmalloc(BINSZ);
    char name[64];
    int n_written = 0;

    pthread_mutex_lock(&g_b_mtx);
    for (;;) {
        while (g_share.msg_n == 0 && !g_proc_done)
            pthread_cond_wait(&g_b_cond, &g_b_mtx);
        if (g_share.overflow) {
            pthread_mutex_unlock(&g_b_mtx);
            fprintf(stderr, "extract_thread: A 置 overflow（B 写盘不及），产物作废——整条重跑\n");
            return NULL;
        }
        if (g_share.msg_n == 0 && g_proc_done) break;
        uint32_t mt = g_share.msg_tail;
        char *m = g_share.msg[mt];
        g_share.msg_tail = (uint32_t)((mt + 1) % SHM_MSG_CAP);
        g_share.msg_n--;
        pthread_mutex_unlock(&g_b_mtx);

        int id, start, end;
        int p = 0;
        if (sscanf(m, "%d %d %d%n", &id, &start, &end, &p) != 3)
            die("extract_thread: 消息解析失败（id/start/end）");
        int k = 0;
        for (;;) {
            int fn; unsigned long long jo, jl, bo, bl;
            int p2;
            if (sscanf(m + p, " %d %llu %llu %llu %llu%n",
                       &fn, &jo, &jl, &bo, &bl, &p2) != 5)
                break;
            p += p2;
            if (jl > jpg_cap) {
                jpg_cap = (size_t)jl;
                jpg_buf = realloc(jpg_buf, jpg_cap);
                if (!jpg_buf) die("extract_thread: jpg 缓冲 realloc 失败");
            }
            ring_read(jo, jpg_buf, (size_t)jl);
            ring_read(bo, bin_buf, BINSZ);
            snprintf(name, sizeof(name), "%s_f%d.jpg", g_vhash, fn);
            write_frame_file(g_frames_dir, name, jpg_buf, (size_t)jl);
            snprintf(name, sizeof(name), "%s_f%d.bin", g_vhash, fn);
            write_frame_file(g_dino_dir, name, bin_buf, BINSZ);
            __atomic_store_n(&g_share.head, (uint32_t)((bo + bl) % g_share.ring_bytes),
                             __ATOMIC_RELEASE);
            k++;
        }
        if (k == 0) die("extract_thread: 消息解析失败（无帧条目）");
        n_written += k;
        pthread_mutex_lock(&g_b_mtx);
    }
    pthread_mutex_unlock(&g_b_mtx);
    printf("extract_thread: 写盘完成，%d 帧（jpg/bin）\n", n_written);
    free(jpg_buf);
    free(bin_buf);
    ck(cuCtxPopCurrent(NULL), "pop ctx extract");
    return NULL;
}

/* shot 定稿 → 共享产物：选中帧 jpg/bin（g_sb 缓冲内）→ 共享显存环（字节流 SPSC）。
 * A 永不等 B：环满/消息满 → overflow 置位后继续（B 写盘跟不上 = 帧文件缺 → 整条重跑）。
 * 数据块布局：[jpg_len u32][jpg][bin BINSZ]，消息："{id} {start} {end} {fn jpg_off jpg_len bin_off bin_len}..." */
static void share_push_shot(int id, int start, int end, const int *fns, int nf) {
    if (!g_share_on) return;
    if (nf <= 0) return;
    uint64_t cap = g_share.ring_bytes;

    int idxs[16];
    uint64_t need = 0;
    for (int k = 0; k < nf; k++) {
        long idx = (long)fns[k] - (long)g_sb.start;
        if (!g_sb.open || idx < 0 || idx >= g_sb.n) {
            fprintf(stderr, "share_push_shot: 帧 %d 不在 shot 缓冲——置 overflow\n", fns[k]);
            __atomic_store_n(&g_share.overflow, 1, __ATOMIC_RELEASE);
            return;
        }
        idxs[k] = (int)idx;
        need += 4 + g_sb.len[idx] + BINSZ;
    }
    uint32_t head = __atomic_load_n(&g_share.head, __ATOMIC_ACQUIRE);
    uint32_t tail = g_share.tail;
    uint64_t used = (uint64_t)((tail + cap - head) % cap);
    if (used + need > cap)
        __atomic_store_n(&g_share.overflow, 1, __ATOMIC_RELEASE);   /* B 消费不及 → 数据作废 */

    uint64_t pos = tail;
    for (int k = 0; k < nf; k++) {
        size_t jl = g_sb.len[idxs[k]];
        uint32_t jl32 = (uint32_t)jl;
        ring_write(pos, &jl32, 4);                     /* 块头：jpg 长度 */
        pos += 4;
        ring_write(pos, g_sb.jpg + g_sb.off[idxs[k]], jl);
        pos += jl;
        ring_write(pos, g_sb.bins + (size_t)idxs[k] * BINSZ, BINSZ);
        pos += BINSZ;
    }
    ck(cuStreamSynchronize(procS), "share sync");   /* 数据写完才发消息（B 读即所得） */
    __sync_synchronize();
    g_share.tail = (uint32_t)(pos % cap);

    /* 定稿消息（B sscanf 解析，零二次约定；mutex 保护 + cond 唤醒 B） */
    pthread_mutex_lock(&g_b_mtx);
    if (g_share.msg_n >= SHM_MSG_CAP)
        __atomic_store_n(&g_share.overflow, 1, __ATOMIC_RELEASE);   /* 消息满 → 作废 */
    int mi = g_share.msg_head;
    char *m = g_share.msg[mi];
    uint64_t p0 = tail;
    int p = snprintf(m, SHM_MSG_LEN, "%d %d %d", id, start, end);
    if (p < 0 || p >= SHM_MSG_LEN - 64) die("share 消息过长");
    for (int k = 0; k < nf; k++) {
        size_t jl = g_sb.len[idxs[k]];
        int r = snprintf(m + p, (size_t)(SHM_MSG_LEN - p), " %d %llu %llu %llu %llu",
                         fns[k], (unsigned long long)((p0 + 4) % cap), (unsigned long long)jl,
                         (unsigned long long)((p0 + 4 + jl) % cap), (unsigned long long)BINSZ);
        if (r < 0 || p + r >= SHM_MSG_LEN) die("share 消息过长");
        p += r;
        p0 += 4 + jl + BINSZ;
    }
    g_share.msg_head = (uint32_t)((mi + 1) % SHM_MSG_CAP);
    g_share.msg_n++;
    pthread_mutex_unlock(&g_b_mtx);
    pthread_cond_broadcast(&g_b_cond);
}

/* 结束清理：A 出全部产物后置 status=0（B 消费完 msg 即退出），拆共享区 */
static void share_destroy(void) {
    if (g_share_dev)  { ck(cuMemFree(g_share_dev), "free share ring"); g_share_dev = 0; }
    g_share_on = 0;
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

/* ═══════════════════════════════════════════════════════
 * 在线状态机（逐位复刻 build_cuts/build_shots 语义）
 *  build_cuts:  过滤 c<=1||c>=total + MIN_GAP=2 相邻去重（丢弃当前，gap<2 不入列）
 *  build_shots: 二次过滤+qsort+完全去重 = 恒等（build_cuts 输出已有序唯一）
 *  → 在线确认 cut c_k 时 finalize shot (k-1)：[start=cf[k-2] 或 0, end=c_k-1]
 *    解码结束 finalize 最后 shot：[start=cf 末尾 或 0, end=total-1]
 *  c>=total 在线恒不触发（c=i-1 < i <= total，越界保护在 display_cb）
 * ═══════════════════════════════════════════════════════ */
static void finalize_shot(int id, int start, int end, int keep_from);

static void on_cut(int c) {
    if (c <= 1) return;                          /* build_cuts: c<=1 skip */
    /* 切点确认唯一 = score>=threshold（照抄 vf_scdet_vulkan.c），无黑帧干预。
     * 黑帧段内 scdet 切点（mafd 突跳）照切；黑段末冗余切点由防空行丢弃
     * （黑帧段确认已把 current_shot_start 推进到段后，边界等价） */
    if (c <= current_shot_start) return;         /* cut 不早于当前 shot 起点（防空 shot） */
    if (g_n_cf > 0 && c - g_cf[g_n_cf - 1] < MIN_GAP) return;   /* 丢弃当前（照抄） */
    /* 确认 cut c 时 finalize shot [g_shot_id] = [start=current_shot_start, end=c-1]
     * （照抄 build_shots：bounds=[0, cuts..., total]，shot i = [bounds[i], bounds[i+1]-1]；
     *  无条件 finalize：cut1 确认即产出 shot 0，否则首 shot 被合并边界错误） */
    finalize_shot(g_shot_id++, current_shot_start, c - 1, c);
    current_shot_start = c;
    g_cf[g_n_cf++] = c;
}

/* shot 结束：选帧（process_shot，照抄）+ 共享产物（选中帧 NV12 池内 →
 * 共享显存环 + 定稿消息，A 不等 B）+ 缓冲平移（保留到 keep_from 帧）。
 * keep_from：下一 shot 起始帧（on_cut 路径 = c；黑帧段确认 = 黑帧段起始；
 * 黑帧段结束 = 段后第一帧；解码结束尾部 finalize 传 -1）。平移前选中帧已从
 * 完整缓冲提取（share_push_shot），黑帧段切分时黑帧帧保留在缓冲供黑帧 shot
 * 定稿提取（2026-08-19 大名）。 */
static void finalize_shot(int id, int start, int end, int keep_from) {
    if (id >= MAX_SHOTS) die("finalize_shot: Shot count exceeded MAX_SHOTS limit");
    if (end < start) die("finalize_shot: 空 shot（状态机错误）");
    Shot shot = {id, start, end};
    process_shot(&shot, &online_reps[id], &online_kfs[id], &online_nkfs[id]);
    sk.shots[id] = shot;
    sk.n_shots = id + 1;

    /* 共享产物：选中帧集合 = kfs + （n_kfs==0 退化时）rep 兜底帧——
     * 与现役 unified_extract 写盘集合逐帧一致（A 编码 jpg/bin → B 写盘即所得） */
    int nk = online_nkfs[id];
    int fns[16], nf = 0;
    if (nk == 0) fns[nf++] = online_reps[id];
    for (int k = 0; k < nk && nf < 16; k++) fns[nf++] = online_kfs[id][k];
    share_push_shot(id, start, end, fns, nf);

    printf("  [shot %d] %d-%d -> rep=%d kfs=%d (buf %.1f MB)\n",
           id, start, end, online_reps[id], nk,
           g_sb.open ? (double)g_sb.jpg_used / (1024 * 1024) : 0.0);
    fflush(stdout);

    /* 缓冲平移：保留 [keep_from, 末尾] 帧，供下一 shot 定稿提取选中帧。
     * cap/jpg_cap 及已分配 off/len/bins/jpg 留着复用。keep_from 不在缓冲
     * （黑帧段结束：段后第一帧尚未 push；尾部 finalize：无下一 shot）→ 清空，
     * 由下一帧 sb_push 重新 open。 */
    if (g_sb.open && keep_from >= g_sb.start && keep_from < g_sb.start + g_sb.n) {
        int keep_idx = keep_from - g_sb.start;
        int num_keep = g_sb.n - keep_idx;
        size_t keep_off = g_sb.off[keep_idx];

        if (g_sb.jpg_used > keep_off) {
            size_t keep_bytes = g_sb.jpg_used - keep_off;
            memmove(g_sb.jpg, g_sb.jpg + keep_off, keep_bytes);
            g_sb.jpg_used = keep_bytes;
        } else {
            g_sb.jpg_used = 0;
        }

        for (int i = 0; i < num_keep; i++) {
            g_sb.off[i] = g_sb.off[keep_idx + i] - keep_off;
            g_sb.len[i] = g_sb.len[keep_idx + i];
        }
        memmove(g_sb.bins, g_sb.bins + (size_t)keep_idx * BINSZ, (size_t)num_keep * BINSZ);

        g_sb.n = num_keep;
        g_sb.start = keep_from;
    } else {
        g_sb.open = 0;
        g_sb.n = 0;
        g_sb.jpg_used = 0;
    }
}

/* 黑帧段独立 shot（2026-08-19 大名，判定照 ffmpeg blackdetect）：
 * blackdetect_eval 确认黑帧段 [seg_start, seg_end]（ratio>=pic_th 持续>=d）
 * 后调用。黑帧段前画面先收尾成普通 shot，黑帧段本身独立成 shot，段后开新
 * 普通 shot。防空：黑帧段不从已 finalize 区回退（scdet 切点已推进
 * current_shot_start 时，黑帧段从 current_shot_start 起，不重叠不空 shot）。 */
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
    if (i >= sk.total_frames) {   /* 越界保护（帧数超骨架声明） */
        nv.cuvidUnmapVideoFrame(g_decoder, dptr64);
        cuCtxPopCurrent(NULL);
        return 1;
    }

    /* 取空闲 slot（满则背压阻塞解码）→ NV12 拷贝到环形缓冲 → 等 GPU 读完表面才能 unmap */
    int slot = q_take_empty();
    upload_nv12_dev(dptr, (size_t)pitch, dIn[slot], inS);
    ck(cuEventRecord(g_in_ready[slot], inS), "record ready");
    ck(cuStreamSynchronize(inS), "sync in");
    nv.cuvidUnmapVideoFrame(g_decoder, dptr64);
    cuCtxPopCurrent(NULL);
    q_put_full(slot, i);
    return 1;
}

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
static void process_frame(int fn, int slot) {
    CUdeviceptr cur = dIn[slot];
    ck(cuStreamWaitEvent(procS, g_in_ready[slot], 0), "wait ready");

    /* ① 提交段（全链到 procS，零同步直到最后；kernel 一字不动，只换流）
     * 抽帧路径照抄现役（1168-1186 行）：work 尺寸 RGB + 224 bin + nvjpeg 编码，
     * 与分析 kernel 同流提交，② 一次 sync 后 jpg/bin 全就绪，再 sb_push 入 shot 缓冲 */
    size_t len = g_jpg_cap;   /* nvjpeg 压缩流长度（② sync 后有效；UE_NO_SHARE 时闲置） */
    if (g_share_on) {
        unsigned long long a1 = (unsigned long long)cur;        /* 环形 NV12：Y@0 + UV@inUVoff（inPitch 对齐） */
        unsigned long long a2 = (unsigned long long)(cur + inUVoff);
        unsigned long long a6 = (unsigned long long)g_rgb_dev;
        int aP = (int)inPitch, aW = work_w, aH = work_h;
        void *kp[6] = { &a1, &aP, &a2, &aW, &aH, &a6 };
        ck(cuLaunchKernel(g_nv12_full, (work_w * work_h + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp, NULL), "nv12_rgb_full");
        unsigned long long a224 = (unsigned long long)g_rgb224_dev;
        void *kp2[6] = { &a1, &aP, &a2, &aW, &aH, &a224 };
        ck(cuLaunchKernel(g_nv12_224, (224 * 224 + 255) / 256, 1, 1, 256, 1, 1, 0, procS, kp2, NULL), "nv12_rgb224");
        ck(cuMemcpyDtoHAsync(g_bin_buf, g_rgb224_dev, BINSZ, procS), "dl rgb224");
        /* nvjpeg 编码（device RGB 输入零 DtoH；同流保证 RGB 就绪，尺寸=work） */
        nvjpegImage_t src; memset(&src, 0, sizeof(src));
        src.channel[0] = (unsigned char *)g_rgb_dev;
        src.pitch[0]   = (size_t)work_w * 3;
        nj.nvjpegEncodeImage(nj.jh, nj.jstate, nj.jparams, &src, NVJPEG_INPUT_RGBI,
                             work_w, work_h, procS);
        nj.nvjpegEncodeRetrieveBitstream(nj.jh, nj.jstate, g_jpg, &len, procS);
    }
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
    {
        static uint32_t hist[256];
        ck(cuMemcpyDtoHAsync(hist, g_q_hist_dev, 256 * 4, procS), "dl qhist");
        ck(cuMemcpyDtoHAsync(&g_q_tout_h, g_q_tout_dev, 4, procS), "dl qtout");
        ck(cuMemcpyDtoHAsync(&g_q_vrep_h, g_q_vrep_dev, 4, procS), "dl qvrep");
        ck(cuMemcpyDtoHAsync(&g_q_brng_h, g_q_brng_dev, 4, procS), "dl qbrng");

        /* GPU 评分小帧：y224_box → y224_lap（灰度 g_small_dev 只作显存中间缓冲；
         * dhash64 已随方案 A 移除——video_hash = 文件 SHA256，不再逐帧算哈希） */
        static unsigned int bsum_h[NB_BLOCKS];
        static long long lap_h[NB_BLOCKS * 2];
        unsigned long long y1 = (unsigned long long)cur;
        int yP = inPitch;
        void *kp_b[6] = { &y1, &yP, &work_w, &work_h, &g_small_dev, &g_bsum_dev };
        void *kp_l[2] = { &g_small_dev, &g_lap_dev };
        ck(cuLaunchKernel(g_ybox, NB_BLOCKS, 1, 1, 256, 1, 1, 0, procS, kp_b, NULL), "y224_box");
        ck(cuLaunchKernel(g_ylap, NB_BLOCKS, 1, 1, 256, 1, 1, 0, procS, kp_l, NULL), "y224_lap");
        ck(cuMemcpyDtoHAsync(bsum_h, g_bsum_dev, (size_t)NB_BLOCKS * 4, procS), "dl bsum");
        ck(cuMemcpyDtoHAsync(lap_h, g_lap_dev, (size_t)NB_BLOCKS * 16, procS), "dl lap");

        /* ② 1 次同步：procS 全链（分析 + DTOH）就绪 */
        ck(cuStreamSynchronize(procS), "sync proc");

        /* ③ 状态机段（CPU，照抄原 display_cb 顺序） */
        /* 先：黑帧段确认（照抄 vf_blackdetect_vulkan.c:evaluate），确认即拆黑帧段独立 shot */
        blackdetect_eval(fn);
        /* 后：scdet 切点（照抄 vf_scdet_vulkan.c：score>=threshold 唯一条件，无黑帧干预）；
         * 黑段末冗余切点由 on_cut 防空行丢弃（黑帧段已推进 current_shot_start，边界等价） */
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
                g_cuts[g_n_cuts++] = fn - 1;   /* 00 对齐：检测帧号 = 当前帧 - 1 */
                on_cut(fn - 1);                /* 在线确认（逐位复刻 build_cuts） */
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

        /* 帧 jpg/bin 入 shot 缓冲（照抄现役；② sync 已保证同流数据就绪） */
        if (g_share_on) sb_push(fn, g_jpg, len, g_bin_buf);
    }

    /* ④ slot 归还：prev 占一槽（引用语义），其余回空闲栈
     * 注意：slot 已成为新 prev，下一帧 ① 段 scdet 还要读它的显存，不能归还——
     * 只还 old_prev，空槽池守恒，背压才有效（否则每帧还 2 取 1 池子膨胀，
     * 队列越界写坏 g_q_slot/g_q_fn）。freeze ref 占槽已随 S10 裁剪移除。 */
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
        /* 解码提前结束：total_frames 截断到实际解码帧数（照抄 Preproc 注释） */
        fprintf(stderr, "unified_extract: 解码到 %d/%d 帧，total_frames 截断为 %d\n",
                decoded, sk.total_frames, decoded);
        sk.total_frames = decoded;
    }
    if (sk.total_frames < 2) die("解码帧数不足 2 帧，无法继续处理");

    /* S9 黑帧墙尾部收尾（照抄 vf_blackdetect_vulkan.c uninit） */
    if (black_start >= 0) {
        int last = decoded - 1; if (last < 0) last = 0;
        if ((last - black_start) >= (long long)(BLACK_DUR_SEC * sk.fps)) {
            if (black_n < MAX_BLACK) {   /* 黑帧墙记录（照抄 report_black_region） */
                black_seg_start[black_n] = black_start;
                black_seg_end[black_n] = last;
                black_n++;
            }
            black_seg_split(black_start, last);   /* 尾部黑帧段独立 shot */
        }
        black_start = -1;
    }

    /* 最后 shot 收尾（照抄 build_shots：bounds 尾 = total_frames；
     * 尾部无下一 shot，keep_from=-1 清空缓冲） */
    if (current_shot_start < sk.total_frames) {
        finalize_shot(g_shot_id++, current_shot_start, sk.total_frames - 1, -1);
    }
    sk.n_shots = g_shot_id;

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
 * 解码 + 分析 + 选帧 + 共享产物（线 A 全速单遍；抽帧归线 B）
 * ═══════════════════════════════════════════════════════ */
static void decode_and_extract(const char *video) {
    /* 短边 960 缩放（横屏固定高，竖屏固定宽；NVDEC 硬件缩放，抽帧/分析共用）
     * 照抄 Preproc.c SCALE_SHORT 语义（领导 2026-08-07 拍板：短边 960 长边按比例） */
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
    gpu_load_cubin("unified_kernels.cubin");
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
    ck(cuMemAlloc(&g_rgb_dev, (size_t)work_w * work_h * 3), "alloc rgb");
    ck(cuMemAlloc(&g_rgb224_dev, (size_t)BINSZ), "alloc rgb224");

    nvcuvid_open();
    nvjpeg_open();
    share_init();   /* 共享显存环 + 共享内存头（UE_NO_SHARE=1 纯分析） */

    /* nvjpeg 编码参数（state/params 各建一次全程复用，绑定 procS 处理线程流，照抄现役） */
    if (nj.nvjpegCreateSimple(&nj.jh) != NVJPEG_STATUS_SUCCESS) die("nvjpegCreateSimple 失败");
    if (nj.nvjpegEncoderStateCreate(nj.jh, &nj.jstate, procS) != NVJPEG_STATUS_SUCCESS) die("EncoderStateCreate 失败");
    if (nj.nvjpegEncoderParamsCreate(nj.jh, &nj.jparams, procS) != NVJPEG_STATUS_SUCCESS) die("EncoderParamsCreate 失败");
    nj.nvjpegEncoderParamsSetQuality(nj.jparams, jpg_quality(), procS);
    nj.nvjpegEncoderParamsSetSamplingFactors(nj.jparams, NVJPEG_CSS_420, procS);   /* 默认 4:4:4 文件大 2-3 倍，必须显式 420 */
    nj.nvjpegEncodeGetBufferSize(nj.jh, nj.jparams, work_w, work_h, &g_jpg_cap);
    g_jpg = xmalloc(g_jpg_cap + 1024);

    sb_init();   /* shot 缓冲（cap=256 / jpg_cap=1MB；0 起点会命中 sb_push nc*=2 死循环） */

    /* 处理线程（照 ffmpeg frame_thread：解码线程产出帧，处理线程消费分析+选帧+保存） */
    if (pthread_create(&g_proc_tid, NULL, proc_thread, NULL) != 0) die("pthread_create 失败");

    /* 线 B 写盘线程（B 本来就跟 A 一起启动，同进程绑定）：等消息 → D2H → 写盘即用即丢 */
    if (g_share_on) {
        if (pthread_create(&g_b_tid, NULL, extract_thread, NULL) != 0) die("pthread_create(B) 失败");
    }

    /* S3 demux C 内化：自写 mov 样本表 + avcC→annexb（抄 h264_mp4toannexb.c）；
     * 全量喂包（分析 + 抽帧都需要每一帧，无提前终止） */
    AnnexB ab; memset(&ab, 0, sizeof(ab));
    if (g_mov.v_extradata_size > 0 && annexb_open(&ab, g_mov.v_extradata, g_mov.v_extradata_size) < 0)
        die("annexb_open（avcC 解析）失败");
    uint8_t *sam = xmalloc(64 * 1024 * 1024);
    int failed = 0;
    for (int i = 0; i < g_mov.v_n && !failed; i++) {
        long n = mov_read_sample(&g_mov, &g_mov.v[i], sam);
        if (n < 0) { failed = 1; break; }
        uint8_t *out = NULL; int osz = 0;
        if (annexb_filter(&ab, sam, (int)n, &out, &osz) < 0) { failed = 1; break; }
        if (out) {
            CUVIDSOURCEDATAPACKET csp; memset(&csp, 0, sizeof(csp));
            csp.payload = out;
            csp.payload_size = (unsigned long)osz;
            if (nv.cuvidParseVideoData(g_parser, &csp) != CUDA_SUCCESS) {
                fprintf(stderr, "unified_extract: cuvidParseVideoData 失败\n");
                failed = 1;
                free(out);
                break;
            }
            free(out);
        }
    }
    annexb_close(&ab);
    free(sam);

    {   /* 流结束 flush 尾部显示帧 */
        CUVIDSOURCEDATAPACKET eos; memset(&eos, 0, sizeof(eos));
        eos.flags = CUVID_PKT_ENDOFSTREAM;
        nv.cuvidParseVideoData(g_parser, &eos);
    }

    if (g_parser)  { nv.cuvidDestroyVideoParser(g_parser);  g_parser  = NULL; }
    if (g_decoder) { nv.cuvidDestroyDecoder(g_decoder);     g_decoder = NULL; }
    if (nv.lib)    { dlclose(nv.lib); nv.lib = NULL; }

    /* 解码结束：通知处理线程收尾（截断/黑帧尾部/冻结尾部/最后 shot 全在 proc_thread 内）；
     * B 写盘线程放行用 g_proc_done（join 后置位，见 extract_thread 注释） */
    g_eos = 1;
    pthread_cond_broadcast(&g_q_has_data);
    pthread_join(g_proc_tid, NULL);
    g_proc_done = 1;                     /* proc 尾消息（末 shot）已全部入队，B 写盘放行 */
    pthread_mutex_lock(&g_b_mtx);
    pthread_cond_broadcast(&g_b_cond);
    pthread_mutex_unlock(&g_b_mtx);
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
    snprintf(g_out_raw,  sizeof(g_out_raw),  "%s/preproc/cuts/%s_cuts.json",
             out_root, g_vhash);
    snprintf(g_out_skel, sizeof(g_out_skel), "%s/preproc/skeleton/%s_skeleton.json",
             out_root, g_vhash);
    snprintf(g_out_sf,   sizeof(g_out_sf),   "%s/preproc/select_frames/%s_select_frames.json",
             out_root, g_vhash);
    snprintf(g_out_events, sizeof(g_out_events), "%s/preproc/events/%s_events.json",
             out_root, g_vhash);
    snprintf(g_out_feat, sizeof(g_out_feat), "%s/preproc/features/%s_features.json",
             out_root, g_vhash);
    snprintf(g_frames_dir, sizeof(g_frames_dir), "%s/preproc/frames", out_root);
    snprintf(g_dino_dir,   sizeof(g_dino_dir),   "%s/preproc/frames224", out_root);

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

    /* video_hash（文件 SHA256 前 6 位）已在输出路径设置前算好，用于 g_vhash 文件名前缀；
     * share_init 发布共享区时一并写 ShareHdr，线 B 从一开始就能用正式名写盘（无需临时名+统一 rename） */

    /* 一次解码：分析 + 在线选帧 + 共享产物（抽帧归线 B） */
    printf("Phase 1/1: decode + analysis + shared output...\n");
    decode_and_extract(video);

    /* 全产出（JSON 照旧协议；帧文件由 B 线程写盘） */
    build_cuts();              /* 写 JSON 前过滤 raw cuts（照抄 Preproc：c<=1/c>=total skip + MIN_GAP） */
    write_raw_cuts(g_out_raw);
    write_skeleton(g_out_skel, vid_name, mode_b_project ? mode_b_project : vid_name);
    write_features(g_out_feat);
    write_events(g_out_events);
    write_output(g_out_sf, online_reps, online_kfs, online_nkfs);

    /* B 线程收尾（g_proc_done 已置位，消费完最后消息即退出）→ 共享环才可释放 */
    if (g_share_on) pthread_join(g_b_tid, NULL);

    /* 清理：共享环 → nvjpeg → CUDA（ctx 还在） */
    share_destroy();
    /* nvjpeg 销毁顺序（官方 sample 语义）：state → params → handle（先 handle 会 segfault） */
    if (nj.jstate) nj.nvjpegEncoderStateDestroy(nj.jstate);
    if (nj.jparams) nj.nvjpegEncoderParamsDestroy(nj.jparams);
    if (nj.jh) nj.nvjpegDestroy(nj.jh);
    if (nj.lib) dlclose(nj.lib);
    free(g_jpg);
    cuda_destroy();

    free(g_mafd); free(g_scores);
    free(g_quality); free(sharpness);
    free(brightness_arr);
    for (int i = 0; i < sk.n_shots; i++) free(online_kfs[i]);
    free(online_kfs); free(online_nkfs); free(online_reps);
    free(sk.shots);
    sb_free();
    mov_close(&g_mov);
    return 0;
}