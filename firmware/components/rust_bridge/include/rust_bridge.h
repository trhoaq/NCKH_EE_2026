#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t sample_rate;
    uint32_t frame_samples;
    uint16_t bits_per_sample;
    uint16_t channels;
    uint32_t wakeword_count;
    uint32_t keyword_set_version;
} rust_bridge_audio_config_t;

typedef struct {
    uint8_t detected;
    uint8_t action_id;
    uint16_t reserved;
    uint32_t score_milli;
    char keyword[32];
} rust_bridge_detection_t;

esp_err_t rust_bridge_init(rust_bridge_audio_config_t *out_audio_config);
esp_err_t rust_bridge_reset_session(void);
esp_err_t rust_bridge_process_samples(const int16_t *samples,
                                      size_t sample_count,
                                      rust_bridge_detection_t *out_detection);
bool rust_bridge_is_ready(void);
size_t rust_bridge_frame_bytes(void);

#ifdef __cplusplus
}
#endif
