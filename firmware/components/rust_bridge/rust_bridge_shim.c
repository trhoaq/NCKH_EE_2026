#include "rust_bridge.h"

#include <stdbool.h>
#include <string.h>

extern int wakeword_bridge_init(rust_bridge_audio_config_t *out_audio_config);
extern int wakeword_bridge_reset_session(void);
extern int wakeword_bridge_process_samples(const int16_t *samples,
                                           size_t sample_count,
                                           rust_bridge_detection_t *out_detection);

static rust_bridge_audio_config_t s_audio_config = {0};
static bool s_bridge_ready = false;

esp_err_t rust_bridge_init(rust_bridge_audio_config_t *out_audio_config) {
    int result = wakeword_bridge_init(&s_audio_config);
    s_bridge_ready = (result == 0 && s_audio_config.wakeword_count > 0);
    if (out_audio_config != NULL) {
        *out_audio_config = s_audio_config;
    }
    return s_bridge_ready ? ESP_OK : ESP_FAIL;
}

esp_err_t rust_bridge_reset_session(void) {
    return wakeword_bridge_reset_session() == 0 ? ESP_OK : ESP_FAIL;
}

esp_err_t rust_bridge_process_samples(const int16_t *samples,
                                      size_t sample_count,
                                      rust_bridge_detection_t *out_detection) {
    if (out_detection != NULL) {
        memset(out_detection, 0, sizeof(*out_detection));
    }
    return wakeword_bridge_process_samples(samples, sample_count, out_detection) == 0 ? ESP_OK : ESP_FAIL;
}

bool rust_bridge_is_ready(void) {
    return s_bridge_ready;
}

size_t rust_bridge_frame_bytes(void) {
    return (size_t) s_audio_config.frame_samples * (size_t) s_audio_config.channels * 2u;
}
