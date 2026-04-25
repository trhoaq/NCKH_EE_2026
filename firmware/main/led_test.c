#include "led_test.h"

#include <stdbool.h>
#include <stdint.h>

#include "esp_log.h"
#include "led_strip.h"

#define LED_STRIP_BLINK_GPIO 48
#define LED_STRIP_LED_NUMBERS 1
#define LED_STRIP_RMT_RES_HZ (10 * 1000 * 1000)
#define LED_BRIGHTNESS 50

static const char *TAG = "led_test";
static led_strip_handle_t s_led_strip;
static bool s_led_ready = false;

static esp_err_t led_test_set_rgb(uint8_t red, uint8_t green, uint8_t blue) {
    if (!s_led_ready) {
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t result = led_strip_set_pixel(s_led_strip, 0, red, green, blue);
    if (result != ESP_OK) {
        return result;
    }
    return led_strip_refresh(s_led_strip);
}

esp_err_t led_test_init(void) {
    if (s_led_ready) {
        return ESP_OK;
    }

    led_strip_config_t strip_config = {
            .strip_gpio_num = LED_STRIP_BLINK_GPIO,
            .max_leds = LED_STRIP_LED_NUMBERS,
            .led_model = LED_MODEL_WS2812,
            .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
    };

    led_strip_rmt_config_t rmt_config = {
            .clk_src = RMT_CLK_SRC_DEFAULT,
            .resolution_hz = LED_STRIP_RMT_RES_HZ,
    };

    esp_err_t result = led_strip_new_rmt_device(&strip_config, &rmt_config, &s_led_strip);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize RGB LED on GPIO %d: %s",
                 LED_STRIP_BLINK_GPIO,
                 esp_err_to_name(result));
        return result;
    }

    s_led_ready = true;
    ESP_LOGI(TAG, "RGB LED initialized on GPIO %d", LED_STRIP_BLINK_GPIO);
    return led_test_set_status(LED_TEST_STATUS_NO_COMMAND);
}

esp_err_t led_test_set_status(led_test_status_t status) {
    switch (status) {
        case LED_TEST_STATUS_ON:
            ESP_LOGI(TAG, "LED status=ON, color=green");
            return led_test_set_rgb(0, LED_BRIGHTNESS, 0);
        case LED_TEST_STATUS_OFF:
            ESP_LOGI(TAG, "LED status=OFF, color=blue");
            return led_test_set_rgb(0, 0, LED_BRIGHTNESS);
        case LED_TEST_STATUS_NO_COMMAND:
        default:
            ESP_LOGI(TAG, "LED status=NO_COMMAND, color=red");
            return led_test_set_rgb(LED_BRIGHTNESS, 0, 0);
    }
}
