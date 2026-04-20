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
#include "rust_bridge.h"

static const char *TAG = "tcp_audio_server";
static rust_bridge_audio_config_t s_audio_config;
static const uint32_t MAX_SESSION_DURATION_MS = 4000;

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
            .sample_rate = htonl(s_audio_config.sample_rate),
            .frame_samples = htonl(s_audio_config.frame_samples),
            .bits_per_sample = htons(s_audio_config.bits_per_sample),
            .channels = htons(s_audio_config.channels),
            .wakeword_count = htonl(s_audio_config.wakeword_count),
            .detector_ready = htonl(rust_bridge_is_ready() ? 1u : 0u),
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

static bool send_detection(int socket_fd,
                           uint32_t session_id,
                           const rust_bridge_detection_t *detection) {
    uint16_t keyword_length = (uint16_t) strnlen(detection->keyword, sizeof(detection->keyword));
    vcp_detection_payload_t payload = {
            .session_id = htonl(session_id),
            .action_id = htons((uint16_t) detection->action_id),
            .keyword_length = htons(keyword_length),
            .score_milli = htonl(detection->score_milli),
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
    return write_exact(socket_fd, detection->keyword, keyword_length);
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
    bool session_started = false;
    uint32_t active_session_id = 0;
    uint8_t *session_buffer = NULL;
    size_t session_capacity_bytes = 0;
    size_t session_size_bytes = 0;

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
            send_error(socket_fd, active_session_id, VCP_ERROR_BAD_PACKET, "invalid magic");
            break;
        }

        if (header.type == VCP_PACKET_SESSION_START) {
            if (header.length != sizeof(vcp_session_start_payload_t)) {
                send_error(socket_fd, active_session_id, VCP_ERROR_BAD_PACKET, "invalid session start payload");
                if (!discard_bytes(socket_fd, header.length)) {
                    break;
                }
                continue;
            }

            vcp_session_start_payload_t payload = {0};
            if (!read_exact(socket_fd, &payload, sizeof(payload))) {
                break;
            }

            uint32_t session_id = ntohl(payload.session_id);
            uint32_t sample_rate = ntohl(payload.sample_rate);
            uint32_t frame_samples = ntohl(payload.frame_samples);
            uint16_t bits_per_sample = ntohs(payload.bits_per_sample);
            uint16_t channels = ntohs(payload.channels);

            if (!rust_bridge_is_ready()) {
                send_error(socket_fd, session_id, VCP_ERROR_DETECTOR_NOT_READY, "rust wakeword engine not ready");
                continue;
            }

            if (sample_rate != s_audio_config.sample_rate
                || frame_samples != s_audio_config.frame_samples
                || bits_per_sample != s_audio_config.bits_per_sample
                || channels != s_audio_config.channels) {
                send_error(socket_fd, session_id, VCP_ERROR_UNSUPPORTED_FORMAT, "audio format mismatch");
                continue;
            }

            size_t required_capacity = ((size_t) sample_rate
                                        * (size_t) channels
                                        * sizeof(int16_t)
                                        * MAX_SESSION_DURATION_MS) / 1000u;
            if (required_capacity == 0) {
                send_error(socket_fd, session_id, VCP_ERROR_INTERNAL, "invalid session capacity");
                continue;
            }
            if (session_capacity_bytes != required_capacity) {
                free(session_buffer);
                session_buffer = (uint8_t *) malloc(required_capacity);
                session_capacity_bytes = session_buffer != NULL ? required_capacity : 0u;
            }
            if (session_buffer == NULL) {
                send_error(socket_fd, session_id, VCP_ERROR_INTERNAL, "session buffer allocation failed");
                continue;
            }

            rust_bridge_reset_session();
            active_session_id = session_id;
            session_started = true;
            session_size_bytes = 0u;
            send_state(socket_fd, active_session_id, VCP_STATE_RECEIVING);
        } else if (header.type == VCP_PACKET_AUDIO_FRAME) {
            if (header.length < sizeof(vcp_audio_frame_prefix_t)) {
                send_error(socket_fd, active_session_id, VCP_ERROR_BAD_PACKET, "audio frame payload too small");
                if (!discard_bytes(socket_fd, header.length)) {
                    break;
                }
                continue;
            }

            vcp_audio_frame_prefix_t prefix = {0};
            if (!read_exact(socket_fd, &prefix, sizeof(prefix))) {
                break;
            }

            uint32_t session_id = ntohl(prefix.session_id);
            uint32_t audio_bytes = header.length - sizeof(prefix);
            size_t expected_bytes = rust_bridge_frame_bytes();
            if (!session_started || session_id != active_session_id) {
                send_error(socket_fd, session_id, VCP_ERROR_BAD_SESSION, "session not active");
                if (!discard_bytes(socket_fd, audio_bytes)) {
                    break;
                }
                continue;
            }

            if (audio_bytes != expected_bytes) {
                send_error(socket_fd, session_id, VCP_ERROR_UNSUPPORTED_FORMAT, "frame byte count mismatch");
                if (!discard_bytes(socket_fd, audio_bytes)) {
                    break;
                }
                continue;
            }

            if (session_size_bytes + audio_bytes > session_capacity_bytes) {
                send_error(socket_fd, session_id, VCP_ERROR_INTERNAL, "session audio exceeds max duration");
                if (!discard_bytes(socket_fd, audio_bytes)) {
                    break;
                }
                continue;
            }

            if (!read_exact(socket_fd, session_buffer + session_size_bytes, audio_bytes)) {
                break;
            }
            session_size_bytes += audio_bytes;
        } else if (header.type == VCP_PACKET_SESSION_END) {
            if (header.length != sizeof(vcp_session_end_payload_t)) {
                send_error(socket_fd, active_session_id, VCP_ERROR_BAD_PACKET, "invalid session end payload");
                if (!discard_bytes(socket_fd, header.length)) {
                    break;
                }
                continue;
            }

            vcp_session_end_payload_t payload = {0};
            if (!read_exact(socket_fd, &payload, sizeof(payload))) {
                break;
            }

            active_session_id = ntohl(payload.session_id);
            if (session_started && session_size_bytes > 0u) {
                rust_bridge_detection_t detection = {0};
                send_state(socket_fd, active_session_id, VCP_STATE_PROCESSING);
                esp_err_t result = rust_bridge_process_samples(
                        (const int16_t *) session_buffer,
                        session_size_bytes / sizeof(int16_t),
                        &detection);
                if (result != ESP_OK) {
                    send_error(socket_fd, active_session_id, VCP_ERROR_INTERNAL, "wakeword processing failed");
                } else if (detection.detected) {
                    send_detection(socket_fd, active_session_id, &detection);
                    command_router_apply_action((vcp_action_id_t) detection.action_id, detection.keyword);
                }
            }
            session_started = false;
            session_size_bytes = 0u;
            rust_bridge_reset_session();
            send_state(socket_fd, active_session_id, VCP_STATE_READY);
        } else {
            send_error(socket_fd, active_session_id, VCP_ERROR_BAD_PACKET, "unsupported packet type");
            if (!discard_bytes(socket_fd, header.length)) {
                break;
            }
        }
    }

    free(session_buffer);
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

esp_err_t tcp_audio_server_start(const rust_bridge_audio_config_t *audio_config) {
    s_audio_config = *audio_config;
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
