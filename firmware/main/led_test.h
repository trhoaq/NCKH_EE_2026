#pragma once

#include "esp_err.h"

typedef enum {
    LED_TEST_STATUS_NO_COMMAND = 0,
    LED_TEST_STATUS_ON = 1,
    LED_TEST_STATUS_OFF = 2,
} led_test_status_t;

esp_err_t led_test_init(void);
esp_err_t led_test_set_status(led_test_status_t status);
