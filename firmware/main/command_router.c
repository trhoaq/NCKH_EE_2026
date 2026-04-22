#include "command_router.h"

#include <stdbool.h>

#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"

static const char *TAG = "command_router";
static const gpio_num_t DEVICE_GPIO = GPIO_NUM_48;
static bool s_device_enabled = false;

void command_router_init(void) {
    s_device_enabled = false;
    gpio_config_t config = {
            .pin_bit_mask = 1ULL << DEVICE_GPIO,
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t result = gpio_config(&config);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure GPIO %d: %s", (int) DEVICE_GPIO, esp_err_to_name(result));
        return;
    }
    gpio_set_level(DEVICE_GPIO, 0);
    ESP_LOGI(TAG, "Command router initialized, default state=OFF");
}

void command_router_apply_action(vcp_action_id_t action_id, const char *keyword) {
    switch (action_id) {
        case VCP_ACTION_TURN_ON:
            s_device_enabled = true;
            gpio_set_level(DEVICE_GPIO, 1);
            ESP_LOGI(TAG, "Command \"%s\" mapped to TURN_ON, device_state=%s",
                     keyword != NULL ? keyword : "unknown",
                     s_device_enabled ? "ON" : "OFF");
            break;
        case VCP_ACTION_TURN_OFF:
            s_device_enabled = false;
            gpio_set_level(DEVICE_GPIO, 0);
            ESP_LOGI(TAG, "Command \"%s\" mapped to TURN_OFF, device_state=%s",
                     keyword != NULL ? keyword : "unknown",
                     s_device_enabled ? "ON" : "OFF");
            break;
        default:
            ESP_LOGW(TAG, "Ignoring unknown action id=%d for keyword \"%s\"",
                     action_id,
                     keyword != NULL ? keyword : "unknown");
            break;
    }
}
