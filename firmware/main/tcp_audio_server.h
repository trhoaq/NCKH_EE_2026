#pragma once

#include "esp_err.h"
#include "rust_bridge.h"

esp_err_t tcp_audio_server_start(const rust_bridge_audio_config_t *audio_config);
