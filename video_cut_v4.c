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

static double probe_fps(const char *video_path) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v quiet -select_streams v:0 "
        "-show_entries stream=avg_frame_rate "
        "-of default=noprint_wrappers=1:nokey=1 '%s'", video_path);
    FILE *fp = popen(cmd, "r");
    if (!fp) return 30.0;
    char buf[64]; fgets(buf, sizeof(buf), fp);
    pclose(fp);
    int num = 30000, den = 1001;
    sscanf(buf, "%d/%d", &num, &den);
    return (double)num / den;
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

    double fps = probe_fps(video_path);
    printf("Video: %s\nShots: %d  FPS: %.4f\n", video_path, n_shots, fps);

    /* ── Build timestamp string (comma-separated, no expr: prefix) ── */
    char *timestamps = malloc(n_shots * 16);
    if (!timestamps) { fprintf(stderr, "OOM\n"); return 1; }
    timestamps[0] = '\0';
    int ts_len = 0;

    for (int i = 1; i < n_shots; i++) {  /* skip shot 0 (starts at 0) */
        double t = shots[i].start / fps;
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "%s%.6f", i > 1 ? "," : "", t);
        if (ts_len + n < n_shots * 16) {
            strcat(timestamps, buf);
            ts_len += n;
        }
    }
    printf("Timestamps: %d chars\n", ts_len);

    /* ── Single ffmpeg: one decode, one encode, segment muxer splits ── */
    char cmd[32768];
    char out_pattern[4096];
    snprintf(out_pattern, sizeof(out_pattern), "%s/segment_%%04d.mp4", out_dir);

    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -v error "
        "-hwaccel cuda -hwaccel_output_format cuda "
        "-i '%s' "
        "-force_key_frames %s "
        "-c:v h264_nvenc -preset p1 -cq 26 -forced-idr 1 "
        "-an "
        "-f segment -segment_times %s -reset_timestamps 1 "
        "'%s'",
        video_path, timestamps, timestamps, out_pattern);

    printf("Encoding...\n");
    int ret = system(cmd);
    free(timestamps);

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
