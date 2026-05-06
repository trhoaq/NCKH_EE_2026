#include "actuator.h"

#include <stdint.h>

#include "driver/mcpwm_prelude.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "SERVO_CONTROL";

#define SERVO_MIN_PULSEWIDTH_US 500
#define SERVO_MAX_PULSEWIDTH_US 2500
#define SERVO_MIN_DEGREE 0
#define SERVO_MAX_DEGREE 180
#define SERVO_DEFAULT_CLOSED_DEGREE 0
#define SERVO_DEFAULT_OPEN_DEGREE 180
#define SERVO_PULSE_GPIO 3
#define SERVO_TIMEBASE_RESOLUTION_HZ 1000000
#define SERVO_PWM_PERIOD_TICKS 20000
#define SERVO_MOVE_DURATION_MS 2000
#define SERVO_MOVE_STEPS 40

static mcpwm_timer_handle_t s_timer = NULL;
static mcpwm_oper_handle_t s_operator = NULL;
static mcpwm_gen_handle_t s_generator = NULL;
static mcpwm_cmpr_handle_t s_comparator = NULL;
static bool s_initialized = false;
static bool s_enabled = false;
static int s_current_angle = SERVO_DEFAULT_CLOSED_DEGREE;

static uint32_t angle_to_compare(int angle) {
    return (uint32_t) ((angle - SERVO_MIN_DEGREE)
            * (SERVO_MAX_PULSEWIDTH_US - SERVO_MIN_PULSEWIDTH_US)
            / (SERVO_MAX_DEGREE - SERVO_MIN_DEGREE)
            + SERVO_MIN_PULSEWIDTH_US);
}

static esp_err_t actuator_set_angle(int angle) {
    if (!s_initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    if (angle < SERVO_MIN_DEGREE || angle > SERVO_MAX_DEGREE) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t result = mcpwm_comparator_set_compare_value(s_comparator, angle_to_compare(angle));
    if (result == ESP_OK) {
        s_current_angle = angle;
    }
    return result;
}

static esp_err_t actuator_move_to_angle(int target_angle) {
    int start_angle = s_current_angle;
    int delta = target_angle - start_angle;
    TickType_t step_delay = pdMS_TO_TICKS(SERVO_MOVE_DURATION_MS / SERVO_MOVE_STEPS);

    ESP_LOGI(TAG, "Moving servo from %d to %d over %dms", start_angle, target_angle, SERVO_MOVE_DURATION_MS);
    for (int step = 1; step <= SERVO_MOVE_STEPS; step++) {
        int angle = start_angle + ((delta * step) / SERVO_MOVE_STEPS);
        esp_err_t result = actuator_set_angle(angle);
        if (result != ESP_OK) {
            return result;
        }
        vTaskDelay(step_delay);
    }

    return ESP_OK;
}

esp_err_t actuator_init(void) {
    if (s_initialized) {
        return ESP_OK;
    }

    mcpwm_timer_config_t timer_config = {
            .group_id = 0,
            .clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT,
            .resolution_hz = SERVO_TIMEBASE_RESOLUTION_HZ,
            .period_ticks = SERVO_PWM_PERIOD_TICKS,
            .count_mode = MCPWM_TIMER_COUNT_MODE_UP,
    };
    ESP_RETURN_ON_ERROR(mcpwm_new_timer(&timer_config, &s_timer), TAG, "Failed to create MCPWM timer");

    mcpwm_operator_config_t operator_config = {
            .group_id = 0,
    };
    ESP_RETURN_ON_ERROR(mcpwm_new_operator(&operator_config, &s_operator), TAG, "Failed to create MCPWM operator");
    ESP_RETURN_ON_ERROR(mcpwm_operator_connect_timer(s_operator, s_timer), TAG, "Failed to connect timer");

    mcpwm_generator_config_t generator_config = {
            .gen_gpio_num = SERVO_PULSE_GPIO,
    };
    ESP_RETURN_ON_ERROR(mcpwm_new_generator(s_operator, &generator_config, &s_generator), TAG, "Failed to create MCPWM generator");

    mcpwm_comparator_config_t comparator_config = {
            .flags.update_cmp_on_tez = true,
    };
    ESP_RETURN_ON_ERROR(mcpwm_new_comparator(s_operator, &comparator_config, &s_comparator), TAG, "Failed to create MCPWM comparator");

    ESP_RETURN_ON_ERROR(
            mcpwm_generator_set_action_on_timer_event(
                    s_generator,
                    MCPWM_GEN_TIMER_EVENT_ACTION(
                            MCPWM_TIMER_DIRECTION_UP,
                            MCPWM_TIMER_EVENT_EMPTY,
                            MCPWM_GEN_ACTION_HIGH)),
            TAG,
            "Failed to configure generator timer action");
    ESP_RETURN_ON_ERROR(
            mcpwm_generator_set_action_on_compare_event(
                    s_generator,
                    MCPWM_GEN_COMPARE_EVENT_ACTION(
                            MCPWM_TIMER_DIRECTION_UP,
                            s_comparator,
                            MCPWM_GEN_ACTION_LOW)),
            TAG,
            "Failed to configure generator compare action");

    ESP_RETURN_ON_ERROR(mcpwm_timer_enable(s_timer), TAG, "Failed to enable MCPWM timer");
    ESP_RETURN_ON_ERROR(
            mcpwm_timer_start_stop(s_timer, MCPWM_TIMER_START_NO_STOP),
            TAG,
            "Failed to start MCPWM timer");

    s_initialized = true;
    s_enabled = false;
    s_current_angle = SERVO_DEFAULT_CLOSED_DEGREE;
    ESP_RETURN_ON_ERROR(actuator_set_angle(SERVO_DEFAULT_CLOSED_DEGREE), TAG, "Failed to set default servo angle");
    ESP_LOGI(TAG, "Servo actuator initialized on GPIO %d", SERVO_PULSE_GPIO);
    return ESP_OK;
}

esp_err_t actuator_set_enabled(bool enabled) {
    if (enabled == s_enabled) {
        ESP_LOGI(TAG, "Servo already %s, skipping movement", s_enabled ? "OPEN" : "CLOSED");
        return ESP_OK;
    }

    int target_angle = enabled ? SERVO_DEFAULT_OPEN_DEGREE : SERVO_DEFAULT_CLOSED_DEGREE;
    esp_err_t result = actuator_move_to_angle(target_angle);
    if (result != ESP_OK) {
        return result;
    }

    s_enabled = enabled;
    ESP_LOGI(TAG, "Servo state=%s", s_enabled ? "OPEN" : "CLOSED");
    return ESP_OK;
}
