#pragma once

#include <stdbool.h>

#include "esp_err.h"

esp_err_t actuator_init(void);
esp_err_t actuator_set_enabled(bool enabled);
