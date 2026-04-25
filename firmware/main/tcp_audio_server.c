#include "tcp_audio_server.h"

#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/socket.h>

#include "command_router.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "protocol.h"

static const char *TAG = "tcp_audio_server";

static bool read_exact(int socket_fd, void *buffer, size_t length) {
    uint8_t *cursor = (uint8_t *) buffer;
    size_t offset = 0;
    while (offset < length) {
        int bytes_read = recv(socket_fd, cursor + offset, length - offset, 0);
        if (bytes_read <= 0) {
            return false;
        }
        offset += (size_t) bytes_read;
    }
    return true;
}

static bool write_exact(int socket_fd, const void *buffer, size_t length) {
    const uint8_t *cursor = (const uint8_t *) buffer;
    size_t offset = 0;
    while (offset < length) {
        int bytes_written = send(socket_fd, cursor + offset, length - offset, 0);
        if (bytes_written <= 0) {
            return false;
        }
        offset += (size_t) bytes_written;
    }
    return true;
}

static bool send_packet(int socket_fd, uint16_t type, const void *payload, uint32_t payload_length) {
    vcp_packet_header_t header = {
            .magic = htonl(VCP_MAGIC),
            .type = htons(type),
            .flags = 0,
            .length = htonl(payload_length),
    };

    if (!write_exact(socket_fd, &header, sizeof(header))) {
        return false;
    }

    if (payload_length == 0) {
        return true;
    }
    return write_exact(socket_fd, payload, payload_length);
}

static bool send_hello(int socket_fd) {
    vcp_hello_payload_t hello = {
            .protocol_version = htonl(VCP_PROTOCOL_VERSION),
            .sample_rate = 0,
            .frame_samples = 0,
            .bits_per_sample = 0,
            .channels = 0,
            .wakeword_count = 0,
            .detector_ready = htonl(1u),
    };
    return send_packet(socket_fd, VCP_PACKET_HELLO, &hello, sizeof(hello));
}

static bool send_state(int socket_fd, uint32_t session_id, vcp_state_code_t state_code) {
    vcp_state_payload_t payload = {
            .session_id = htonl(session_id),
            .state_code = htonl((uint32_t) state_code),
    };
    return send_packet(socket_fd, VCP_PACKET_STATE, &payload, sizeof(payload));
}

static bool send_error(int socket_fd,
                       uint32_t session_id,
                       vcp_error_code_t error_code,
                       const char *message) {
    const char *safe_message = message != NULL ? message : "";
    uint16_t message_length = (uint16_t) strnlen(safe_message, 96);
    vcp_error_payload_t payload = {
            .session_id = htonl(session_id),
            .error_code = htons((uint16_t) error_code),
            .message_length = htons(message_length),
    };

    vcp_packet_header_t header = {
            .magic = htonl(VCP_MAGIC),
            .type = htons(VCP_PACKET_ERROR),
            .flags = 0,
            .length = htonl(sizeof(payload) + message_length),
    };

    if (!write_exact(socket_fd, &header, sizeof(header))) {
        return false;
    }
    if (!write_exact(socket_fd, &payload, sizeof(payload))) {
        return false;
    }
    if (message_length == 0) {
        return true;
    }
    return write_exact(socket_fd, safe_message, message_length);
}

static bool send_detection_packet(int socket_fd,
                                  uint32_t session_id,
                                  uint16_t action_id,
                                  const char *keyword,
                                  uint32_t score_milli) {
    const char *safe_keyword = keyword != NULL ? keyword : "";
    uint16_t keyword_length = (uint16_t) strnlen(safe_keyword, 31);
    vcp_detection_payload_t payload = {
            .session_id = htonl(session_id),
            .action_id = htons(action_id),
            .keyword_length = htons(keyword_length),
            .score_milli = htonl(score_milli),
    };

    vcp_packet_header_t header = {
            .magic = htonl(VCP_MAGIC),
            .type = htons(VCP_PACKET_DETECTION),
            .flags = 0,
            .length = htonl(sizeof(payload) + keyword_length),
    };

    if (!write_exact(socket_fd, &header, sizeof(header))) {
        return false;
    }
    if (!write_exact(socket_fd, &payload, sizeof(payload))) {
        return false;
    }
    if (keyword_length == 0) {
        return true;
    }
    return write_exact(socket_fd, safe_keyword, keyword_length);
}

static bool is_supported_action(uint16_t action_id) {
    return action_id == VCP_ACTION_NONE
           || action_id == VCP_ACTION_TURN_ON
           || action_id == VCP_ACTION_TURN_OFF;
}

static bool discard_bytes(int socket_fd, uint32_t length) {
    uint8_t buffer[128];
    uint32_t remaining = length;
    while (remaining > 0) {
        size_t chunk = MIN(sizeof(buffer), remaining);
        if (!read_exact(socket_fd, buffer, chunk)) {
            return false;
        }
        remaining -= chunk;
    }
    return true;
}

static void handle_client(int socket_fd) {
    uint32_t active_request_id = 0;

    if (!send_hello(socket_fd)) {
        ESP_LOGW(TAG, "Failed to send HELLO packet");
        return;
    }

    while (true) {
        vcp_packet_header_t network_header = {0};
        if (!read_exact(socket_fd, &network_header, sizeof(network_header))) {
            break;
        }

        vcp_packet_header_t header = {
                .magic = ntohl(network_header.magic),
                .type = ntohs(network_header.type),
                .flags = ntohs(network_header.flags),
                .length = ntohl(network_header.length),
        };

        if (header.magic != VCP_MAGIC) {
            send_error(socket_fd, active_request_id, VCP_ERROR_BAD_PACKET, "invalid magic");
            break;
        }

        if (header.type == VCP_PACKET_COMMAND) {
            if (header.length < sizeof(vcp_command_payload_t)) {
                send_error(socket_fd, active_request_id, VCP_ERROR_BAD_PACKET, "command payload too small");
                if (!discard_bytes(socket_fd, header.length)) {
                    break;
                }
                continue;
            }

            vcp_command_payload_t payload = {0};
            if (!read_exact(socket_fd, &payload, sizeof(payload))) {
                break;
            }

            uint32_t request_id = ntohl(payload.request_id);
            uint16_t action_id = ntohs(payload.action_id);
            uint16_t keyword_length = ntohs(payload.keyword_length);
            uint32_t score_milli = ntohl(payload.score_milli);
            uint32_t remaining_bytes = header.length - sizeof(payload);
            uint32_t bytes_to_read = MIN((uint32_t) keyword_length, remaining_bytes);
            char keyword[32] = {0};

            if (bytes_to_read > 0) {
                if (!read_exact(socket_fd, keyword, bytes_to_read)) {
                    break;
                }
                keyword[MIN(bytes_to_read, (uint32_t) sizeof(keyword) - 1u)] = '\0';
            }
            if (remaining_bytes > bytes_to_read) {
                if (!discard_bytes(socket_fd, remaining_bytes - bytes_to_read)) {
                    break;
                }
            }

            if (!is_supported_action(action_id)) {
                send_error(socket_fd, request_id, VCP_ERROR_UNSUPPORTED_ACTION, "unsupported action id");
                continue;
            }

            active_request_id = request_id;
            send_state(socket_fd, request_id, VCP_STATE_PROCESSING);
            command_router_apply_action((vcp_action_id_t) action_id, keyword);
            send_detection_packet(socket_fd, request_id, action_id, keyword, score_milli);
            send_state(socket_fd, request_id, VCP_STATE_READY);
        } else if (header.type == VCP_PACKET_SESSION_START
                   || header.type == VCP_PACKET_AUDIO_FRAME
                   || header.type == VCP_PACKET_SESSION_END) {
            send_error(socket_fd, active_request_id, VCP_ERROR_BAD_PACKET, "audio streaming is disabled in command mode");
            if (!discard_bytes(socket_fd, header.length)) {
                break;
            }
        } else {
            send_error(socket_fd, active_request_id, VCP_ERROR_BAD_PACKET, "unsupported packet type");
            if (!discard_bytes(socket_fd, header.length)) {
                break;
            }
        }
    }
}

static void tcp_audio_server_task(void *arg) {
    int listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (listen_socket < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno=%d", errno);
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in server_address = {0};
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = htonl(INADDR_ANY);
    server_address.sin_port = htons(VCP_DEFAULT_PORT);

    int reuse_flag = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &reuse_flag, sizeof(reuse_flag));

    if (bind(listen_socket, (struct sockaddr *) &server_address, sizeof(server_address)) != 0) {
        ESP_LOGE(TAG, "Socket bind failed: errno=%d", errno);
        close(listen_socket);
        vTaskDelete(NULL);
        return;
    }

    if (listen(listen_socket, 1) != 0) {
        ESP_LOGE(TAG, "Socket listen failed: errno=%d", errno);
        close(listen_socket);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "TCP audio server listening on port %d", VCP_DEFAULT_PORT);

    while (true) {
        struct sockaddr_in client_address = {0};
        socklen_t client_address_length = sizeof(client_address);
        int client_socket = accept(listen_socket, (struct sockaddr *) &client_address, &client_address_length);
        if (client_socket < 0) {
            ESP_LOGE(TAG, "Accept failed: errno=%d", errno);
            continue;
        }

        ESP_LOGI(TAG, "Client connected: %s", inet_ntoa(client_address.sin_addr));
        handle_client(client_socket);
        shutdown(client_socket, 0);
        close(client_socket);
        ESP_LOGI(TAG, "Client disconnected");
    }
}

esp_err_t tcp_audio_server_start(void) {
    BaseType_t task_created = xTaskCreatePinnedToCore(
            tcp_audio_server_task,
            "tcp_audio_server",
            8192,
            NULL,
            5,
            NULL,
            tskNO_AFFINITY);
    return task_created == pdPASS ? ESP_OK : ESP_FAIL;
}
