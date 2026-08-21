/* preproc/unified_kernels.cu — 一次解码出全分辨率关键帧（unified_extract）全部 CUDA kernel 合并版
 *
 * 用法: nvcc -arch=all -fatbin -o unified_kernels.cubin unified_kernels.cu（编译生成 .cubin 供 duo_analyze 加载）
 * 依赖: CUDA 工具链（nvcc）；合并来源：preproc_kernels.cu 全文（scdet_sad/blackdetect_count/
 *       freeze_sad/vm_conv_y/vm_conv_x/vm_sad/q_hist/q_tout/q_vrep/q_brng/nv12_rgb224/
 *       y224_box/y224_lap/dhash64/frame_mse）+ fullres_extract_kernels.cu 的 nv12_rgb_full
 *       （追加；nv12_rgb224 已在 preproc_kernels.cu，不重复定义）
 * 产物: unified_kernels.cubin
 */

/* ═══════════ 1. scdet_sad（参考自 FFmpeg 9.0 源码 libavfilter/vulkan/scdet.comp.glsl + vf_scdet_vulkan.c）═══════
 * 算法（YUV 输入，planes=1）：
 *   kernel: Σ|Y_cur − Y_prev|（整数，无符号 8bit）→ 全帧 SAD
 *   CPU:    mafd = SAD × 100.0 / (w×h×255)
 *           score = min(mafd, |mafd − prev_mafd|)，clip 0-100
 *           score >= threshold → 切变
 * 与 vulkan 版逐位一致的关键：SAD 是整数运算，无浮点实现差异。 */

/* 每像素 |cur − prev| 累加（Y 平面，pitch 步长）→ 原子加到 out（先清零） */
extern "C" __global__ void scdet_sad(const unsigned char *__restrict__ prev,
                                     const unsigned char *__restrict__ cur,
                                     int pitch, unsigned long long *__restrict__ out,
                                     int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int col = idx % w, row = idx / w;
    int a = prev[(size_t)row * pitch + col];
    int b = cur[(size_t)row * pitch + col];
    int d = a > b ? a - b : b - a;
    atomicAdd(out, (unsigned long long)d);
}

/* ═══════════ 2. blackdetect_count（参考自 FFmpeg 9.0 源码 libavfilter/vulkan/blackdetect.comp.glsl
 *                                             + libavfilter/vf_blackdetect_vulkan.c）═══════
 * 算法（Y 平面，8bit uint8）：
 *   kernel: 逐像素 y <= thr 计数（glsl: subgroupBallot(in_bounds && value <= threshold)）
 *           → atomicAdd 归约（glsl 16-slice 归约的总数等价，整数加法 exactly-associative）
 *   CPU:    ratio = nb_black / (w×h)；ratio >= pic_th(0.98) 记黑段开始（帧号），
 *           断帧记结束，duration >= d(2.0s) 才输出（宿主状态机照抄 vf_blackdetect_vulkan.c:evaluate）
 * 阈值换算（vf_blackdetect_vulkan.c:191-200，TV range 8bit）：
 *   threshold = (pix_th × (ymax−ymin) + ymin) / imax = (0.10×219+16)/255 ≈ 0.1486
 *   归一化比较 value ≤ threshold ⇔ 整数 y ≤ 37（37/255≈0.1451≤0.1486；38/255≈0.1490>0.1486） */

/* 每像素 y <= thr 计数（Y 平面，pitch 步长）→ 原子加到 out（先清零） */
extern "C" __global__ void blackdetect_count(const unsigned char *__restrict__ y,
                                             int pitch, int w, int h, int thr,
                                             unsigned int *__restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;          /* glsl in_bounds：越界不算黑 */
    int col = idx % w, row = idx / w;
    if (y[(size_t)row * pitch + col] <= thr) /* glsl: value <= threshold */
        atomicAdd(out, 1U);                  /* glsl: subgroupBallot 计数等价 */
}

/* ═══════════ 3. freeze_sad（参考自 FFmpeg 9.0 源码 libavfilter/vf_freezedetect.c + scene_sad.c）═══════
 * 算法（NV12 输入）：
 *   kernel: Σ|ref − cur| 全平面（Y w×h + UV 交错 w×(h/2)，逐字节）
 *           → ffmpeg is_frozen: plane0=Y + plane1=UV 交错（NV12 单平面）逐字节 SAD 同义
 *   CPU:    mafd = sad / count / (1<<bitdepth)   （count = w×h + w×(h/2) = 1.5×w×h）
 *           mafd <= noise(0.001) → 冻结（照抄 is_frozen）
 *           状态机照抄 activate：reference 只在非冻结时更新（冻结段第一帧）；
 *           duration >= d(2.0s) 才输出 freeze_start（= reference 帧）/ freeze_end（= 断帧）
 * 参数（filters.texi 默认）：n/noise=0.001、d/duration=2.0s */

/* 全平面 SAD（Y w×h + UV w×(h/2) 逐字节 |a−b|）→ 原子加到 out（先清零）
 * 索引布局：idx < Y_N(=w×h) 处理 Y 平面；Y_N <= idx < Y_N+UV_N(=Y_N/2) 处理 UV 平面
 * 归约与 scdet_sad 同风格：per-thread atomicAdd（整数加法 exactly-associative） */
extern "C" __global__ void freeze_sad(const unsigned char *__restrict__ ref_y,
                                      const unsigned char *__restrict__ cur_y,
                                      int pitch, int w, int h,
                                      const unsigned char *__restrict__ ref_uv,
                                      const unsigned char *__restrict__ cur_uv,
                                      unsigned long long *__restrict__ out) {
    long long y_n = (long long)w * h;
    long long total = y_n + y_n / 2;            /* 全平面字节数（Y + UV 交错） */
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int col, row, a, b;
    if (idx < y_n) {                            /* Y 平面 */
        col = (int)(idx % w); row = (int)(idx / w);
        a = ref_y[(size_t)row * pitch + col];
        b = cur_y[(size_t)row * pitch + col];
    } else {                                    /* UV 交错平面（宽 = w 字节） */
        long long k = idx - y_n;
        col = (int)(k % w); row = (int)(k / w);
        a = ref_uv[(size_t)row * pitch + col];
        b = cur_uv[(size_t)row * pitch + col];
    }
    int d = a > b ? a - b : b - a;
    atomicAdd(out, (unsigned long long)d);
}

/* ═══════════ 4. vmafmotion（参考自 FFmpeg 9.0 源码 libavfilter/vf_vmafmotion.c）═══════
 * 算法（Y 平面，8bit）：
 *   conv_y: 5-tap 垂直定点 FIR → uint16 temp（>>8，照抄 convolution_y_8bit 的 bits=8）
 *   conv_x: 5-tap 水平定点 FIR → uint16 blur（>>15，照抄 convolution_x 的 BIT_SHIFT）
 *   sad:    Σ|blur_prev − blur_cur|（uint16 逐像素）
 *   CPU:    score = sad / (w×h×2^7)（BIT_SHIFT−8 = 7，输出恒归一化到 8bit）；
 *           首帧 score=0（照抄 ff_vmafmotion_process nb_frames==0 分支）
 * 定点系数：FILTER_5 × 2^15 经 lrint（round-half-even）：
 *   [0.054488685, 0.244201342, 0.402619947, 0.244201342, 0.054488685] → [1785, 8002, 13193, 8002, 1785]
 * 边界（照抄 convolution_x/convolution_y）：j_tap = |j−radius+k|；若 >= w → w−(j_tap−w+1)
 * 显存布局：temp/blur 均 w 紧密 uint16（stride 只影响内存布局，取值序列与 ffmpeg 一致） */

#define VM_RADIUS 2
#define VM_FILT_W 5
#define VM_SHIFT   15   /* conv_x 右移（照抄 convolution_x: sum >> BIT_SHIFT=15） */
#define VM_SHIFT_Y  8   /* conv_y 8bit 输入右移（照抄 convolution_y_8bit: sum >> bits=8，勿用 15！） */

__constant__ int vm_filter[VM_FILT_W] = { 1785, 8002, 13193, 8002, 1785 };

/* 垂直 5-tap：src uint8 Y（pitch 步长）→ dst uint16（w 紧密） */
extern "C" __global__ void vm_conv_y(const unsigned char *__restrict__ src, int pitch,
                                     unsigned short *__restrict__ dst, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int x = idx % w, y = idx / w;
    int sum = 0;
#pragma unroll
    for (int k = 0; k < VM_FILT_W; k++) {
        int i_tap = abs(y - VM_RADIUS + k);          /* FFABS */
        if (i_tap >= h) i_tap = h - (i_tap - h + 1); /* 镜像边界（照抄） */
        sum += vm_filter[k] * src[(size_t)i_tap * pitch + x];
    }
    dst[idx] = (unsigned short)(sum >> VM_SHIFT_Y);
}

/* 水平 5-tap：src uint16（w 紧密）→ dst uint16（w 紧密） */
extern "C" __global__ void vm_conv_x(const unsigned short *__restrict__ src,
                                     unsigned short *__restrict__ dst, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int x = idx % w, y = idx / w;
    const unsigned short *row = src + (size_t)y * w;
    int sum = 0;
#pragma unroll
    for (int k = 0; k < VM_FILT_W; k++) {
        int j_tap = abs(x - VM_RADIUS + k);          /* FFABS */
        if (j_tap >= w) j_tap = w - (j_tap - w + 1); /* 镜像边界（照抄） */
        sum += vm_filter[k] * row[j_tap];
    }
    dst[idx] = (unsigned short)(sum >> VM_SHIFT);
}

/* 帧对 blur SAD（uint16 逐像素 |a−b|）→ 原子加到 out（先清零） */
extern "C" __global__ void vm_sad(const unsigned short *__restrict__ a,
                                  const unsigned short *__restrict__ b,
                                  int n, unsigned long long *__restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int x = a[idx], y = b[idx];
    int d = x > y ? x - y : y - x;
    atomicAdd(out, (unsigned long long)d);
}

/* ═══════════ 5. quality（参考自 FFmpeg 9.0 源码 libavfilter/vf_signalstats.c
 *                                        + libavfilter/vf_entropy.c）═══════
 * 指标（Y 平面 + NV12 交错 UV）：
 *   q_hist:  Y 直方图 256 bin → CPU: YAVG=Σh·v/fs、YLOW/YHIGH=10%/90% 百分位、
 *            YMIN/YMAX=非零最值、entropy=Σ−log2(p)·p（照抄 vf_entropy.c mode=0）
 *   q_tout:  TOUT 像素计数，照抄 FILTER3：outlier(x,y,z)=((|x−y|+|z−y|)/2)−|z−x|>4；
 *            y∈[1,h−1)、x∈[1,w−1)；y−2≥0&&y+2<h 时 FILTER3(2)&&FILTER3(1)，否则 FILTER3(1)
 *   q_vrep:  VREP 行计数，照抄：y∈[4,h)：Σ|p[y−4]−p[y]| < w 记重复行
 *   q_brng:  BRNG 像素计数，照抄：luma<16||luma>235||chromau<16||chromau>240||chromav<16||chromav>240
 *            （NV12：u=uv[2·xc]，v=uv[2·xc+1]，xc=x>>1，yc=y>>1，4:2:0）
 * 归一化（照抄 vf_signalstats.c SET_META）：YAVG/TOUT/BRNG = 值/fs；
 *   VREP = 行数×w/fs = 行数/h（filter8_vrep 返回 score×w） */

/* Y 直方图 256 bin → 原子加到 out（先清零 256×4 字节） */
extern "C" __global__ void q_hist(const unsigned char *__restrict__ y, int pitch,
                                  int w, int h, unsigned int *__restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int col = idx % w, row = idx / w;
    atomicAdd(&out[y[(size_t)row * pitch + col]], 1U);
}

/* TOUT：temporal outlier 像素计数（照抄 filter8_tout + FILTER3 + filter_tout_outlier）
 * 边界照抄：y∈[1,h−1) continue + for (x = 1; x < w - 1; x++)（x 0/w−1 不参与） */
extern "C" __global__ void q_tout(const unsigned char *__restrict__ y, int pitch,
                                  int w, int h, unsigned int *__restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int x = idx % w, yy = idx / w;
    if (yy - 1 < 0 || yy + 1 >= h) return;   /* 照抄 y 边界 continue */
    if (x < 1 || x >= w - 1) return;         /* 照抄 for (x = 1; x < w - 1; x++) */
    int filt;
    if (yy - 2 >= 0 && yy + 2 < h) {         /* 照抄：深 3 行可用时双查 */
        filt = 1;
        for (int j = 1; j <= 2 && filt; j++)
            for (int i = -1; i <= 1 && filt; i++) {
                int a = y[(size_t)(yy - j) * pitch + x + i];
                int b = y[(size_t)yy * pitch + x + i];
                int c = y[(size_t)(yy + j) * pitch + x + i];
                int d1 = a > b ? a - b : b - a;
                int d2 = c > b ? c - b : b - c;
                int d3 = a > c ? a - c : c - a;
                filt = ((d1 + d2) / 2) - d3 > 4;   /* 照抄 filter_tout_outlier */
            }
    } else {
        filt = 1;
        for (int i = -1; i <= 1 && filt; i++) {
            int a = y[(size_t)(yy - 1) * pitch + x + i];
            int b = y[(size_t)yy * pitch + x + i];
            int c = y[(size_t)(yy + 1) * pitch + x + i];
            int d1 = a > b ? a - b : b - a;
            int d2 = c > b ? c - b : b - c;
            int d3 = a > c ? a - c : c - a;
            filt = ((d1 + d2) / 2) - d3 > 4;
        }
    }
    if (filt) atomicAdd(out, 1U);
}

/* VREP：垂直重复行计数（照抄 filter8_vrep：行 y 与 y−4 的逐像素差分和 < w） */
extern "C" __global__ void q_vrep(const unsigned char *__restrict__ y, int pitch,
                                  int w, int h, unsigned int *__restrict__ out) {
    int yrow = blockIdx.x * blockDim.x + threadIdx.x;
    if (yrow < 4 || yrow >= h) return;          /* VREP_START=4 */
    int totdiff = 0;
    for (int x = 0; x < w; x++) {
        int a = y[(size_t)(yrow - 4) * pitch + x];
        int b = y[(size_t)yrow * pitch + x];
        totdiff += a > b ? a - b : b - a;
    }
    if (totdiff < w) atomicAdd(out, 1U);       /* filt = totdiff < w */
}

/* BRNG：广播范围外像素计数（照抄 filter8_brng；NV12 UV 交错 u=偶字节 v=奇字节） */
extern "C" __global__ void q_brng(const unsigned char *__restrict__ y, int pitch,
                                  const unsigned char *__restrict__ uv, int w, int h,
                                  unsigned int *__restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int x = idx % w, yy = idx / w;
    int xc = x >> 1, yc = yy >> 1;              /* hsub=vsub=1（4:2:0） */
    int luma = y[(size_t)yy * pitch + x];
    int chromau = uv[(size_t)yc * pitch + 2 * xc];
    int chromav = uv[(size_t)yc * pitch + 2 * xc + 1];
    if (luma < 16 || luma > 235 || chromau < 16 || chromau > 240 || chromav < 16 || chromav > 240)
        atomicAdd(out, 1U);
}

/* NV12 → RGB8 224×224（DINO 输入真彩色；BT.601 limited-range 定点 4096，整数运算跨卡逐位一致）
 * 每输出像素整数最近邻采样源帧；照抄 libswscale yuv2rgb 系数（1.164/1.596/0.391/0.813/2.018） */
extern "C" __global__ void nv12_rgb224(const unsigned char *__restrict__ y, int pitch,
                                       const unsigned char *__restrict__ uv, int w, int h,
                                       unsigned char *__restrict__ rgb) {
    const int OW = 224, OH = 224;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OW * OH) return;
    int x = idx % OW, yy = idx / OW;
    int sx = (x * w) / OW, sy = (yy * h) / OH;   /* 整数最近邻，确定性 */
    int xc = sx >> 1, yc = sy >> 1;
    int Y = y[(size_t)sy * pitch + sx];
    int U = uv[(size_t)yc * pitch + 2 * xc];
    int V = uv[(size_t)yc * pitch + 2 * xc + 1];
    int c0 = (Y - 16) * 4769;                                        /* 1.164×4096 */
    int R = (c0 + (V - 128) * 6538) >> 12;                           /* +1.596 */
    int G = (c0 - (U - 128) * 1602 - (V - 128) * 3330) >> 12;        /* -0.391 -0.813 */
    int B = (c0 + (U - 128) * 8266) >> 12;                           /* +2.018 */
    unsigned char *o = &rgb[(size_t)yy * OW * 3 + x * 3];
    o[0] = R < 0 ? 0 : (R > 255 ? 255 : (unsigned char)R);
    o[1] = G < 0 ? 0 : (G > 255 ? 255 : (unsigned char)G);
    o[2] = B < 0 ? 0 : (B > 255 ? 255 : (unsigned char)B);
}

/* ═══════════ 6. y224_box（参考自 Preproc.c 原 CPU box_filter：积分图 box 均值）
 * 整数 box 平均：每输出像素 = 源区块和 / 区块面积（整数除法，与 CPU 版逐位一致）
 * 每线程 1 输出像素(224×224)；block 内归约亮度局部和 → bsum_blocks[NB_BLOCKS]
 * CPU 端 196 次加法得全帧亮度和（标量，非像素） */
extern "C" __global__ void y224_box(const unsigned char *__restrict__ y, int pitch, int w, int h,
                                     unsigned char *__restrict__ small, unsigned int *__restrict__ bsum_blocks) {
    const int OW = 224, OH = 224;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int v = 0;
    if (idx < OW * OH) {
        int ox = idx % OW, oy = idx / OW;
        int ys = (oy * h) / OH, ye = ((oy + 1) * h) / OH;
        if (ye <= ys) ye = ys + 1;
        int xs = (ox * w) / OW, xe = ((ox + 1) * w) / OW;
        if (xe <= xs) xe = xs + 1;
        unsigned int s = 0;
        for (int yy = ys; yy < ye; yy++) {
            const unsigned char *r = y + (size_t)yy * pitch + xs;
            for (int xx = 0; xx < xe - xs; xx++) s += r[xx];
        }
        unsigned int cnt = (unsigned int)(ye - ys) * (xe - xs);
        v = (unsigned char)(s / cnt);
        small[idx] = v;
    }
    __shared__ unsigned int sh[256];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) bsum_blocks[blockIdx.x] = sh[0];
}

/* ═══════════ 7. y224_lap（参考自 Preproc.c 原 CPU laplacian_var：Laplacian 方差）
 * 整数拉普拉斯累加（sum/sum2），block 归约 → lap_blocks[block*2]=sum, [block*2+1]=sum2
 * CPU 端：mean=sum/cnt; var=sum2/cnt−mean²（除法在 CPU，跨卡一致） */
extern "C" __global__ void y224_lap(const unsigned char *__restrict__ img,
                                     long long *__restrict__ lap_blocks) {
    const int OW = 224, OH = 224;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % OW, y = idx / OW;
    long long l = 0;
    if (idx < OW * OH && x > 0 && y > 0 && x < OW - 1 && y < OH - 1) {
        l = (long long)img[idx] * 4 - img[idx - 1] - img[idx + 1] - img[idx - OW] - img[idx + OW];
    }
    __shared__ long long sh1[256], sh2[256];
    sh1[threadIdx.x] = l;
    sh2[threadIdx.x] = l * l;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sh1[threadIdx.x] += sh1[threadIdx.x + st];
            sh2[threadIdx.x] += sh2[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { lap_blocks[blockIdx.x * 2] = sh1[0]; lap_blocks[blockIdx.x * 2 + 1] = sh2[0]; }
}

/* ═══════════ 8. dhash64（参考自 Preproc.c 原 CPU compute_fingerprint：8×8 块均值 → 64bit）
 * 每帧 1 block(64 线程)：8×8 块均值（28×28/块，整数除法）→ 行内相邻 56bit + 首/末行 8bit */
extern "C" __global__ void dhash64(const unsigned char *__restrict__ img, unsigned long long *__restrict__ out) {
    const int OW = 224;
    __shared__ unsigned char grid[64];
    int t = threadIdx.x;
    if (t < 64) {
        int oy = t / 8, ox = t % 8;
        long long s = 0;
        for (int yy = oy * 28; yy < (oy + 1) * 28; yy++) {
            const unsigned char *r = img + (size_t)yy * OW + ox * 28;
            for (int xx = 0; xx < 28; xx++) s += r[xx];
        }
        grid[t] = (unsigned char)(s / 784);
    }
    __syncthreads();
    if (t == 0) {
        unsigned long long d = 0;
        for (int oy = 0; oy < 8; oy++)
            for (int ox = 0; ox < 7; ox++)
                d = (d << 1) | (grid[oy * 8 + ox] > grid[oy * 8 + ox + 1]);
        for (int ox = 0; ox < 8; ox++)
            d = (d << 1) | (grid[ox] > grid[56 + ox]);
        *out = d;
    }
}

/* ═══════════ 9. frame_mse（参考自 Preproc.c 原 CPU frame_mse：同 shot 候选帧去重）
 * 全组合对批量：每对 1 block，块内归约 Σ(d²)；out[p] = 总/像素数（浮点除法同 CPU 语义）
 * pairs 为 uniq 帧索引对（消费端查表用） */
extern "C" __global__ void frame_mse(const unsigned char *__restrict__ frames,
                                     const int *__restrict__ pairs, float *__restrict__ out, int npairs) {
    const int NPIX = 224 * 224;
    int p = blockIdx.x;
    long long sum = 0;
    if (p < npairs) {
        int a = pairs[2 * p], b = pairs[2 * p + 1];
        const unsigned char *pa = frames + (size_t)a * NPIX;
        const unsigned char *pb = frames + (size_t)b * NPIX;
        for (int i = threadIdx.x; i < NPIX; i += blockDim.x) {
            int d = (int)pa[i] - pb[i];
            sum += (long long)d * d;
        }
    }
    __shared__ long long sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[p] = (float)((double)sh[0] / (double)NPIX);
}

/* ═══════════ 10. nv12_rgb_full（照抄 fullres_extract_kernels.cu — 全分辨率 NV12 → RGB 1:1）
 * BT.601 limited-range 定点 4096，整数运算跨 GPU 逐位一致 */
extern "C" __global__ void nv12_rgb_full(const unsigned char *__restrict__ y, int pitch,
                                         const unsigned char *__restrict__ uv, int w, int h,
                                         unsigned char *__restrict__ rgb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)w * h) return;
    int x = idx % w, yy = idx / w;
    int xc = x >> 1, yc = yy >> 1;              /* hsub=vsub=1（4:2:0） */
    int Y = y[(size_t)yy * pitch + x];
    int U = uv[(size_t)yc * pitch + 2 * xc];
    int V = uv[(size_t)yc * pitch + 2 * xc + 1];
    int c0 = (Y - 16) * 4769;                                        /* 1.164×4096 */
    int R = (c0 + (V - 128) * 6538) >> 12;                           /* +1.596 */
    int G = (c0 - (U - 128) * 1602 - (V - 128) * 3330) >> 12;        /* -0.391 -0.813 */
    int B = (c0 + (U - 128) * 8266) >> 12;                           /* +2.018 */
    unsigned char *o = &rgb[(size_t)yy * w * 3 + x * 3];
    o[0] = R < 0 ? 0 : (R > 255 ? 255 : (unsigned char)R);
    o[1] = G < 0 ? 0 : (G > 255 ? 255 : (unsigned char)G);
    o[2] = B < 0 ? 0 : (B > 255 ? 255 : (unsigned char)B);
}
