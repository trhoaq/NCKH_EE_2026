#include "command_router.h"

#include <stdbool.h>

#include "esp_log.h"

static const char *TAG = "command_router";
static bool s_device_enabled = false;

void command_router_init(void) {
    s_device_enabled = false;
    ESP_LOGI(TAG, "Command router initialized, default state=OFF");
}

void command_router_apply_action(vcp_action_id_t action_id, const char *keyword) {
    switch (action_id) {
        case VCP_ACTION_TURN_ON:
            s_device_enabled = true;
            ESP_LOGI(TAG, "Wakeword \"%s\" mapped to TURN_ON, device_state=%s",
                     keyword != NULL ? keyword : "unknown",
                     s_device_enabled ? "ON" : "OFF");
            break;
        case VCP_ACTION_TURN_OFF:
            s_device_enabled = false;
            ESP_LOGI(TAG, "Wakeword \"%s\" mapped to TURN_OFF, device_state=%s",
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
