/*
 * video_cut_v4.c — segment muxer: single-pass GPU, zero -ss, frame-accurate
 *
 * 用法: ./video_cut_v4 <skeleton.json>（ffmpeg 一次解码 → force IDR @每段边界 →
 *       NVENC 一次编码 → segment muxer 切分）
 * 依赖: 输入 <skeleton.json>（含 video 路径 + shots 起止帧）；ffmpeg/ffprobe + CUDA NVENC（h264_nvenc）
 * 产物: <skeleton.json 所在目录>/segment_%04d.mp4（逐段 ffprobe 校验帧数）
 *
 * 编译: gcc -o video_cut video_cut_v4.c -O2 -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── JSON ── */
typedef struct { int id, start, end; } Shot;

static char *read_file(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    *len = ftell(fp); rewind(fp);
    char *buf = malloc(*len + 1);
    if (!buf) { fclose(fp); return NULL; }
    fread(buf, 1, *len, fp); fclose(fp);
    buf[*len] = '\0'; return buf;
}

static int extract_str(const char *j, const char *k, char *out, int sz) {
    char s[64]; snprintf(s, sizeof(s), "\"%s\"", k);
    const char *p = strstr(j, s); if (!p) return -1;
    p = strchr(p, ':'); if (!p) return -1;
    p++; while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return -1; p++;
    int i = 0;
    while (*p && *p != '"' && i < sz - 1) {
        if (*p == '\\' && *(p+1)) p++; out[i++] = *p++;
    }
    out[i] = '\0'; return 0;
}

static int parse_shots(const char *j, Shot *s, int max) {
    const char *p = strstr(j, "\"shots\""); if (!p) return 0;
    p = strchr(p, '['); if (!p) return 0; p++;
    int n = 0;
    while (*p && *p != ']' && n < max) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p != '{') break;
        int d = 0; const char *e = p;
        while (*e) { if (*e == '{') d++; if (*e == '}') { d--; if (d == 0) break; } e++; }
        if (!*e) break;
        int olen = (int)(e - p + 1); char obj[4096];
        if (olen >= (int)sizeof(obj)) { p = e + 1; continue; }
        strncpy(obj, p, olen); obj[olen] = '\0';
        int id = -1, st = -1, ed = -1;
        char *k;
        k = strstr(obj, "\"id\"");    if (k) { k = strchr(k,':'); if (k) id = (int)strtol(k+1,NULL,10); }
        k = strstr(obj, "\"start\""); if (k) { k = strchr(k,':'); if (k) st = (int)strtol(k+1,NULL,10); }
        k = strstr(obj, "\"end\"");   if (k) { k = strchr(k,':'); if (k) ed = (int)strtol(k+1,NULL,10); }
        if (id >= 0 && st >= 0 && ed >= 0 && ed >= st) s[n++] = (Shot){id, st, ed};
        p = e + 1;
    }
    return n;
}

static int shot_cmp(const void *a, const void *b) {
    return ((const Shot *)a)->start - ((const Shot *)b)->start;
}

/* 平衡括号树构造 force_key_frames 帧号表达式。
 * 关键：199 项 eq(n,X) 若用左结合 + 会成深度 ~199，超 ffmpeg MAX_DEPTH(100)。
 * 用平衡二分 (a+b)+(c+d) 把深度压到 log2(n)，任意段数都不超限制。
 * 语义：eq(n,X) 在帧 n==X 时为 1，+ 相加非零即强制该帧为关键帧。 */
static void build_bal_expr(char *buf, size_t bufsz, const int *starts, int lo, int hi) {
    if (lo == hi) {
        snprintf(buf, bufsz, "eq(n,%d)", starts[lo]);
        return;
    }
    int mid = (lo + hi) / 2;
    char left[16384], right[16384];
    build_bal_expr(left,  sizeof(left),  starts, lo, mid);
    build_bal_expr(right, sizeof(right), starts, mid + 1, hi);
    snprintf(buf, bufsz, "(%s+%s)", left, right);
}

int main(int argc, char *argv[]) {
    if (argc != 2) { fprintf(stderr, "Usage: %s <skeleton.json>\n", argv[0]); return 1; }

    long json_len;
    char *json = read_file(argv[1], &json_len);
    if (!json) { fprintf(stderr, "Cannot read JSON\n"); return 1; }

    char video_path[4096] = {0};
    extract_str(json, "video", video_path, sizeof(video_path));

    Shot shots[65536];
    int n_shots = parse_shots(json, shots, 65536);
    free(json);
    if (n_shots == 0) { fprintf(stderr, "No shots\n"); return 1; }
    qsort(shots, n_shots, sizeof(Shot), shot_cmp);

    char out_dir[4096];
    { const char *s = strrchr(argv[1], '/');
      if (s) { int d = (int)(s - argv[1]); strncpy(out_dir, argv[1], d); out_dir[d]='\0'; }
      else strcpy(out_dir, "."); }

    printf("Video: %s\nShots: %d\n", video_path, n_shots);

    /* ── 官方组合：-force_key_frames expr:eq(n,X)（强制帧号关键帧，变量 n=帧号）
     *        + -f segment -segment_frames <结束帧>（按帧号精确切分）─
     * 帧号驱动，零 fps 秒换算，不抽搐。 */
    int *starts = malloc((size_t)n_shots * sizeof(int));
    if (!starts) { fprintf(stderr, "OOM\n"); return 1; }
    for (int i = 0; i < n_shots; i++) starts[i] = shots[i].start;
    char frame_expr[65536];
    build_bal_expr(frame_expr, sizeof(frame_expr), starts, 0, n_shots - 1);
    free(starts);
    printf("Force key frames @ abs frame ids (no fps)\n");

    /* segment_frames: 每段结束帧号（官方：到该帧号切分新段） */
    size_t sf_len = 0;
    char *sf = malloc((size_t)n_shots * 16);
    if (!sf) { fprintf(stderr, "OOM\n"); return 1; }
    sf[0] = '\0';
    for (int i = 0; i < n_shots - 1; i++) {
        int n = snprintf(sf + sf_len, (size_t)(n_shots * 16) - sf_len,
                         "%s%d", i > 0 ? "," : "", shots[i].end);
        if (n > 0) sf_len += (size_t)n;
    }

    char cmd[32768];
    char out_pattern[4096];
    snprintf(out_pattern, sizeof(out_pattern), "%s/segment_%%04d.mp4", out_dir);

    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -v error "
        "-hwaccel cuda -hwaccel_output_format cuda "
        "-i '%s' "
        "-force_key_frames 'expr:%s' "
        "-c:v h264_nvenc -preset p1 -cq 26 -forced-idr 1 "
        "-f segment -segment_frames %s -reset_timestamps 1 "
        "'%s'",
        video_path, frame_expr, sf, out_pattern);

    printf("Encoding...\n");
    int ret = system(cmd);
    free(sf);

    if (ret != 0) {
        fprintf(stderr, "ffmpeg failed (exit %d)\n", ret);
        return 1;
    }

    /* ── Verify each segment ── */
    int ok = 0, fail = 0;
    for (int i = 0; i < n_shots; i++) {
        int n_frames = shots[i].end - shots[i].start + 1;
        char seg_path[4096];
        snprintf(seg_path, sizeof(seg_path), "%s/segment_%04d.mp4", out_dir, i);

        char vfy[4096];
        snprintf(vfy, sizeof(vfy),
            "ffprobe -v quiet -count_frames -select_streams v:0 "
            "-show_entries stream=nb_read_frames "
            "-of csv=p=0 '%s'", seg_path);
        FILE *fp = popen(vfy, "r");
        int got = -1;
        if (fp) { fscanf(fp, "%d", &got); pclose(fp); }

        if (got == n_frames) {
            printf("  [%3d] segment_%04d.mp4  %4dfr  OK\n", i, i, n_frames);
            ok++;
        } else {
            printf("  [%3d] segment_%04d.mp4  got=%d exp=%d  ERR\n", i, i, got, n_frames);
            fail++;
        }
        fflush(stdout);
    }

    printf("\nDone: %d OK  %d FAIL  (total %d)\n", ok, fail, n_shots);
    return fail > 0 ? 1 : 0;
}
