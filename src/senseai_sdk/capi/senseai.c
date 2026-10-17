/* Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
 *
 * senseai.c — the C client of the SDK's Python worker. See senseai.h and shim.py.
 * Shared-memory layout is defined once, in shim.py (SHM_LAYOUT); this file mirrors it.
 */
#define _GNU_SOURCE
#include "senseai.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define SHM_MAGIC 0x53414931u  /* "SAI1" */
#define EVENT_SLOTS 64
#define EVENT_SLOT_BYTES 528   /* int64 t + 32 type + 480 text + 8 pad */

/* byte offsets, as in shim.py */
#define OFF_MAGIC 0
#define OFF_VERSION 4
#define OFF_SEQ 8
#define OFF_CMD 16            /* 48 bytes: see pack_command() in shim.py */
#define OFF_EVT_HEAD 64       /* uint64: total events written */
#define OFF_EVT_RING 72

struct senseai_handle {
    pid_t pid;
    FILE *to_worker;
    FILE *from_worker;
    char shm_name[128];
    uint8_t *shm;
    size_t shm_size;
    uint64_t events_read;
    int running;
};

static const char *k_errors[] = {
    "ok", "invalid argument", "could not start the Python worker", "protocol error from the worker",
    "the worker rejected the config, licence or artefacts", "end of replay", "no command yet",
    "shared memory could not be mapped", "call not allowed in this state",
};

const char *senseai_strerror(int code) {
    if (code < 0) code = -code;
    if (code >= (int)(sizeof(k_errors) / sizeof(k_errors[0]))) return "unknown error";
    return k_errors[code];
}

static int read_line(senseai_handle_t *h, char *buf, size_t n) {
    if (!fgets(buf, (int)n, h->from_worker)) return SENSEAI_EPROTO;
    size_t len = strlen(buf);
    while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = 0;
    return SENSEAI_OK;
}

static int send_cmd(senseai_handle_t *h, const char *line, char *reply, size_t n) {
    if (fprintf(h->to_worker, "%s\n", line) < 0 || fflush(h->to_worker) != 0) return SENSEAI_EPROTO;
    return read_line(h, reply, n);
}

int senseai_init(const char *config_path, senseai_handle_t **out) {
    if (!config_path || !out) return SENSEAI_EINVAL;
    int to[2], from[2];
    if (pipe(to) != 0 || pipe(from) != 0) return SENSEAI_ESPAWN;
    pid_t pid = fork();
    if (pid < 0) return SENSEAI_ESPAWN;
    if (pid == 0) {
        dup2(to[0], STDIN_FILENO);
        dup2(from[1], STDOUT_FILENO);
        close(to[1]);
        close(from[0]);
        const char *py = getenv("SENSEAI_PYTHON");
        if (!py) py = "python3";
        execlp(py, py, "-m", "senseai_sdk.capi.shim", "--config", config_path, (char *)NULL);
        _exit(127);
    }
    close(to[0]);
    close(from[1]);
    senseai_handle_t *h = calloc(1, sizeof(*h));
    if (!h) return SENSEAI_ESPAWN;
    h->pid = pid;
    h->to_worker = fdopen(to[1], "w");
    h->from_worker = fdopen(from[0], "r");
    char line[512];
    if (read_line(h, line, sizeof line) != SENSEAI_OK) { senseai_shutdown(h); return SENSEAI_ESPAWN; }
    if (strncmp(line, "ERROR ", 6) == 0) { fprintf(stderr, "senseai: %s\n", line + 6); senseai_shutdown(h); return SENSEAI_ECONFIG; }
    unsigned long size = 0;
    if (sscanf(line, "READY %127s %lu", h->shm_name, &size) != 2) { senseai_shutdown(h); return SENSEAI_EPROTO; }
    char path[160];
    snprintf(path, sizeof path, "/%s", h->shm_name);
    int fd = shm_open(path, O_RDONLY, 0);
    if (fd < 0) { senseai_shutdown(h); return SENSEAI_ESHM; }
    h->shm = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (h->shm == MAP_FAILED) { h->shm = NULL; senseai_shutdown(h); return SENSEAI_ESHM; }
    h->shm_size = size;
    uint32_t magic;
    memcpy(&magic, h->shm + OFF_MAGIC, 4);
    if (magic != SHM_MAGIC) { senseai_shutdown(h); return SENSEAI_EPROTO; }
    *out = h;
    return SENSEAI_OK;
}

int senseai_load_profile(senseai_handle_t *h, const char *name, int version) {
    if (!h || !name) return SENSEAI_EINVAL;
    char line[256], reply[512];
    snprintf(line, sizeof line, "PROFILE %s %d", name, version);
    int rc = send_cmd(h, line, reply, sizeof reply);
    if (rc) return rc;
    if (strncmp(reply, "OK", 2) == 0) return SENSEAI_OK;
    fprintf(stderr, "senseai: %s\n", reply);
    return SENSEAI_ECONFIG;
}

int senseai_step(senseai_handle_t *h) {
    if (!h) return SENSEAI_EINVAL;
    if (h->running) return SENSEAI_ESTATE;
    char reply[512];
    int rc = send_cmd(h, "STEP", reply, sizeof reply);
    if (rc) return rc;
    if (strncmp(reply, "OK", 2) == 0) return SENSEAI_OK;
    if (strncmp(reply, "END", 3) == 0) return SENSEAI_EEND;
    fprintf(stderr, "senseai: %s\n", reply);
    return SENSEAI_EPROTO;
}

int senseai_run(senseai_handle_t *h) {
    if (!h) return SENSEAI_EINVAL;
    char reply[512];
    int rc = send_cmd(h, "RUN", reply, sizeof reply);
    if (rc) return rc;
    if (strncmp(reply, "OK", 2) != 0) return SENSEAI_EPROTO;
    h->running = 1;
    return SENSEAI_OK;
}

int senseai_get_command(senseai_handle_t *h, senseai_command_t *out) {
    if (!h || !out || !h->shm) return SENSEAI_EINVAL;
    for (int attempt = 0; attempt < 1000; attempt++) {
        uint64_t s1, s2;
        memcpy(&s1, h->shm + OFF_SEQ, 8);
        if (s1 & 1) continue;                   /* writer in progress */
        __sync_synchronize();
        uint8_t raw[48];
        memcpy(raw, h->shm + OFF_CMD, sizeof raw);
        __sync_synchronize();
        memcpy(&s2, h->shm + OFF_SEQ, 8);
        if (s1 != s2) continue;
        if (s1 == 0) return SENSEAI_ENOCMD;
        out->seq = s1 / 2;
        memcpy(&out->t_master_ns, raw + 0, 8);
        memcpy(&out->action, raw + 8, 4);
        memcpy(&out->speed_scale, raw + 12, 4);
        memcpy(&out->lateral_offset_m, raw + 16, 4);
        out->stop = raw[20];
        out->hold = raw[21];
        out->latched = raw[22];
        out->degraded = raw[23];
        memcpy(&out->risk, raw + 24, 4);
        memcpy(&out->threshold, raw + 28, 4);
        return SENSEAI_OK;
    }
    return SENSEAI_EPROTO;
}

int senseai_poll_events(senseai_handle_t *h, senseai_event_t *buf, int n) {
    if (!h || !buf || n <= 0 || !h->shm) return -SENSEAI_EINVAL;
    uint64_t head;
    memcpy(&head, h->shm + OFF_EVT_HEAD, 8);
    if (head > h->events_read + EVENT_SLOTS) h->events_read = head - EVENT_SLOTS;  /* overrun: skip */
    int count = 0;
    while (h->events_read < head && count < n) {
        const uint8_t *slot = h->shm + OFF_EVT_RING + (h->events_read % EVENT_SLOTS) * EVENT_SLOT_BYTES;
        memcpy(&buf[count].t_master_ns, slot, 8);
        memcpy(buf[count].type, slot + 8, 32);
        buf[count].type[31] = 0;
        memcpy(buf[count].text, slot + 40, SENSEAI_EVENT_TEXT);
        buf[count].text[SENSEAI_EVENT_TEXT - 1] = 0;
        h->events_read++;
        count++;
    }
    return count;
}

int senseai_health(senseai_handle_t *h, char *buf, size_t size) {
    if (!h || !buf || size == 0) return SENSEAI_EINVAL;
    char reply[8192];
    int rc = send_cmd(h, "HEALTH", reply, sizeof reply);
    if (rc) return rc;
    strncpy(buf, reply, size - 1);
    buf[size - 1] = 0;
    return SENSEAI_OK;
}

int senseai_shutdown(senseai_handle_t *h) {
    if (!h) return SENSEAI_EINVAL;
    if (h->to_worker) {
        char reply[256];
        send_cmd(h, "QUIT", reply, sizeof reply);
        fclose(h->to_worker);
    }
    if (h->from_worker) fclose(h->from_worker);
    if (h->shm) munmap(h->shm, h->shm_size);
    if (h->pid > 0) {
        int status;
        waitpid(h->pid, &status, 0);
    }
    free(h);
    return SENSEAI_OK;
}
