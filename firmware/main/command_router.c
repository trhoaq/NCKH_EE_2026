#include "command_router.h"

#include <stdbool.h>

#include "esp_err.h"
#include "esp_log.h"
#include "led_test.h"

static const char *TAG = "command_router";
static bool s_device_enabled = false;

void command_router_init(void) {
    s_device_enabled = false;
    esp_err_t result = led_test_init();
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize RGB status LED: %s", esp_err_to_name(result));
        return;
    }
    ESP_LOGI(TAG, "Command router initialized, default status=NO_COMMAND");
}

void command_router_apply_action(vcp_action_id_t action_id, const char *keyword) {
    switch (action_id) {
        case VCP_ACTION_TURN_ON:
            s_device_enabled = true;
            led_test_set_status(LED_TEST_STATUS_ON);
            ESP_LOGI(TAG, "Command \"%s\" mapped to TURN_ON, device_state=%s",
                     keyword != NULL ? keyword : "unknown",
                     s_device_enabled ? "ON" : "OFF");
            break;
        case VCP_ACTION_TURN_OFF:
            s_device_enabled = false;
            led_test_set_status(LED_TEST_STATUS_OFF);
            ESP_LOGI(TAG, "Command \"%s\" mapped to TURN_OFF, device_state=%s",
                     keyword != NULL ? keyword : "unknown",
                     s_device_enabled ? "ON" : "OFF");
            break;
        case VCP_ACTION_NONE:
            s_device_enabled = false;
            led_test_set_status(LED_TEST_STATUS_NO_COMMAND);
            ESP_LOGI(TAG, "Command \"%s\" mapped to NO_COMMAND, device_state=%s",
                     keyword != NULL ? keyword : "unknown",
                     s_device_enabled ? "ON" : "OFF");
            break;
        default:
            led_test_set_status(LED_TEST_STATUS_NO_COMMAND);
            ESP_LOGW(TAG, "Ignoring unknown action id=%d for keyword \"%s\"",
                     action_id,
                     keyword != NULL ? keyword : "unknown");
            break;
    }
}
