/* Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
 *
 * example.c — drive the SDK from C: init, load a profile, step, read commands and events.
 *
 *   cc example.c senseai.c -lrt -o example && ./example senseai.yaml
 */
#include "senseai.h"

#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s senseai.yaml [ticks]\n", argv[0]);
        return 2;
    }
    int ticks = argc > 2 ? atoi(argv[2]) : 30;
    senseai_handle_t *h = NULL;
    int rc = senseai_init(argv[1], &h);
    if (rc != SENSEAI_OK) {
        fprintf(stderr, "init failed: %s\n", senseai_strerror(rc));
        return 1;
    }
    rc = senseai_load_profile(h, "null", 0);
    if (rc != SENSEAI_OK) fprintf(stderr, "load_profile: %s\n", senseai_strerror(rc));

    int stops = 0, steps = 0;
    for (int i = 0; i < ticks; i++) {
        rc = senseai_step(h);
        if (rc == SENSEAI_EEND) break;
        if (rc != SENSEAI_OK) { fprintf(stderr, "step: %s\n", senseai_strerror(rc)); break; }
        steps++;
        senseai_command_t cmd;
        if (senseai_get_command(h, &cmd) == SENSEAI_OK) {
            if (cmd.stop) stops++;
            if (i < 3) printf("tick %llu action %d speed_scale %.2f stop %d degraded %d\n",
                              (unsigned long long)cmd.seq, cmd.action, cmd.speed_scale, cmd.stop, cmd.degraded);
        }
        senseai_event_t ev[8];
        int n = senseai_poll_events(h, ev, 8);
        for (int k = 0; k < n; k++) printf("event %s: %s\n", ev[k].type, ev[k].text);
    }
    char health[4096];
    if (senseai_health(h, health, sizeof health) == SENSEAI_OK) printf("health: %.120s...\n", health);
    printf("%d steps, %d with stop requested\n", steps, stops);
    senseai_shutdown(h);
    return 0;
}
