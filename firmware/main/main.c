#include "command_router.h"
#include "esp_err.h"
#include "esp_log.h"
#include "nvs_flash.h"
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

    ESP_LOGI(TAG, "Starting in command receiver mode; phone-side inference is expected");
    ESP_ERROR_CHECK(tcp_audio_server_start());
}
