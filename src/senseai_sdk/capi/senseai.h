/* Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
 *
 * senseai.h — the C API of the SenseAI SDK.
 *
 * A thin client over a long-running Python worker (python3 -m senseai_sdk.capi.shim).
 * senseai_init() starts the worker; control goes over pipes, and the latest reflex command
 * plus a ring of recent events live in a POSIX shared-memory segment the worker writes and
 * this client reads (seqlock: an even sequence number means the record is stable).
 *
 * Every function returns an int error code (SENSEAI_OK == 0); senseai_strerror() describes it.
 * The handle is opaque. Build: cc example.c senseai.c -lrt -o example
 */
#ifndef SENSEAI_H
#define SENSEAI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct senseai_handle senseai_handle_t;

enum senseai_error {
    SENSEAI_OK = 0,
    SENSEAI_EINVAL = 1,      /* bad argument / null pointer */
    SENSEAI_ESPAWN = 2,      /* could not start the Python worker */
    SENSEAI_EPROTO = 3,      /* unexpected reply from the worker */
    SENSEAI_ECONFIG = 4,     /* the worker rejected the config / licence / artefacts */
    SENSEAI_EEND = 5,        /* end of replay: no more ticks */
    SENSEAI_ENOCMD = 6,      /* no command has been produced yet */
    SENSEAI_ESHM = 7,        /* shared memory could not be mapped */
    SENSEAI_ESTATE = 8       /* call not allowed in this state (e.g. step while running) */
};

enum senseai_action {
    SENSEAI_ACTION_NONE = 0,
    SENSEAI_ACTION_SLOW = 1,
    SENSEAI_ACTION_BRAKE = 2,
    SENSEAI_ACTION_SWERVE = 3,
    SENSEAI_ACTION_RETRACT = 4,
    SENSEAI_ACTION_HOLD = 5
};

/* The reflex command: a bounded modifier on the planner's command, never a path. */
typedef struct senseai_command {
    uint64_t seq;             /* increments once per tick */
    int64_t t_master_ns;      /* master clock */
    int32_t action;           /* enum senseai_action */
    float speed_scale;        /* 0..1 multiplier on the planner's speed */
    float lateral_offset_m;   /* signed, bounded by the safety envelope */
    uint8_t stop;             /* controlled stop requested */
    uint8_t hold;             /* freeze manipulator / stay put */
    uint8_t latched;          /* inside the minimum hold */
    uint8_t degraded;         /* produced by the watchdog, not the reflex */
    float risk;
    float threshold;
} senseai_command_t;

#define SENSEAI_EVENT_TEXT 480

typedef struct senseai_event {
    int64_t t_master_ns;
    char type[32];                    /* e.g. "track.new", "reflex.fire", "system.heartbeat" */
    char text[SENSEAI_EVENT_TEXT];    /* one sentence, NUL-terminated */
} senseai_event_t;

/* Start the worker on config_path (a senseai.yaml). Verifies licence and artefacts. */
int senseai_init(const char *config_path, senseai_handle_t **out);

/* Hot-swap the profile; version 0 means the newest the store has. */
int senseai_load_profile(senseai_handle_t *h, const char *name, int version);

/* Single-loop mode: run one tick. Returns SENSEAI_EEND at the end of a replay. */
int senseai_step(senseai_handle_t *h);

/* Threaded mode: let the worker free-run; senseai_get_command() then returns the newest. */
int senseai_run(senseai_handle_t *h);

/* The newest command (copied out; consistent under the seqlock). */
int senseai_get_command(senseai_handle_t *h, senseai_command_t *out);

/* Drain up to n new events into buf; returns the count (>= 0) or a negative error. */
int senseai_poll_events(senseai_handle_t *h, senseai_event_t *buf, int n);

/* The health JSON document (as SenseAI.health().to_json()), NUL-terminated, truncated to size. */
int senseai_health(senseai_handle_t *h, char *buf, size_t size);

/* Stop the worker (final hold on the bridge, usage metered) and free the handle. */
int senseai_shutdown(senseai_handle_t *h);

const char *senseai_strerror(int code);

#ifdef __cplusplus
}
#endif
#endif /* SENSEAI_H */
