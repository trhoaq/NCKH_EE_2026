#include "wifi_softap.h"

#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "protocol.h"

static const char *TAG = "wifi_softap";
static const char *SOFTAP_SSID = "VoiceWakewordESP";
static const char *SOFTAP_PASSWORD = "wakeword123";
static const uint8_t SOFTAP_CHANNEL = 1;
static const uint8_t SOFTAP_MAX_CONNECTIONS = 2;

static void wifi_event_handler(void *arg,
                               esp_event_base_t event_base,
                               int32_t event_id,
                               void *event_data) {
    if (event_base != WIFI_EVENT) {
        return;
    }

    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *event = (wifi_event_ap_staconnected_t *) event_data;
        ESP_LOGI(TAG, "Station joined: " MACSTR ", aid=%d", MAC2STR(event->mac), event->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *event = (wifi_event_ap_stadisconnected_t *) event_data;
        ESP_LOGI(TAG, "Station left: " MACSTR ", aid=%d", MAC2STR(event->mac), event->aid);
    }
}

esp_err_t app_wifi_softap_start(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *ap_netif = esp_netif_create_default_wifi_ap();
    if (ap_netif == NULL) {
        ESP_LOGE(TAG, "Could not create default Wi-Fi AP netif");
        return ESP_FAIL;
    }

    wifi_init_config_t wifi_init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init_config));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
            WIFI_EVENT,
            ESP_EVENT_ANY_ID,
            &wifi_event_handler,
            NULL,
            NULL));

    wifi_config_t wifi_config = {0};
    memcpy(wifi_config.ap.ssid, SOFTAP_SSID, strlen(SOFTAP_SSID));
    memcpy(wifi_config.ap.password, SOFTAP_PASSWORD, strlen(SOFTAP_PASSWORD));
    wifi_config.ap.ssid_len = strlen(SOFTAP_SSID);
    wifi_config.ap.channel = SOFTAP_CHANNEL;
    wifi_config.ap.max_connection = SOFTAP_MAX_CONNECTIONS;
    wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_config.ap.pmf_cfg.required = false;

    if (strlen(SOFTAP_PASSWORD) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    esp_netif_ip_info_t ip_info = {0};
    esp_err_t ip_result = esp_netif_get_ip_info(ap_netif, &ip_info);
    if (ip_result == ESP_OK) {
        ESP_LOGI(TAG,
                 "SoftAP ready: SSID=%s password=%s host=" IPSTR " port=%d",
                 SOFTAP_SSID,
                 SOFTAP_PASSWORD,
                 IP2STR(&ip_info.ip),
                 VCP_DEFAULT_PORT);
    } else {
        ESP_LOGW(TAG,
                 "SoftAP ready but could not read IP info: %s; port=%d",
                 esp_err_to_name(ip_result),
                 VCP_DEFAULT_PORT);
        ESP_LOGI(TAG, "SoftAP ready: SSID=%s password=%s", SOFTAP_SSID, SOFTAP_PASSWORD);
    }
    return ESP_OK;
}
