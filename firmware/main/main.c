#include "command_router.h"
#include "esp_err.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "rust_bridge.h"
#include "tcp_audio_server.h"
#include "wifi_softap.h"

static const char *TAG = "app_main";

void app_main(void) {
    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES || result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        result = nvs_flash_init();
    }
    ESP_ERROR_CHECK(result);

    command_router_init();
    ESP_ERROR_CHECK(wifi_softap_start());

    rust_bridge_audio_config_t audio_config = {0};
    result = rust_bridge_init(&audio_config);
    if (result != ESP_OK) {
        ESP_LOGW(TAG, "Rust wakeword bridge started without active models (err=%s)", esp_err_to_name(result));
    } else {
        ESP_LOGI(TAG,
                 "Wakeword bridge ready: sample_rate=%lu frame_samples=%lu models=%lu",
                 (unsigned long) audio_config.sample_rate,
                 (unsigned long) audio_config.frame_samples,
                 (unsigned long) audio_config.wakeword_count);
    }

    ESP_ERROR_CHECK(tcp_audio_server_start(&audio_config));
}
