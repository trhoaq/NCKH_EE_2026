#pragma once

#include <stdint.h>

#define VCP_PROTOCOL_VERSION 1u
#define VCP_MAGIC 0x56435031u
#define VCP_DEFAULT_PORT 3333

typedef enum {
    VCP_PACKET_HELLO = 1,
    VCP_PACKET_SESSION_START = 2,
    VCP_PACKET_AUDIO_FRAME = 3,
    VCP_PACKET_SESSION_END = 4,
    VCP_PACKET_STATE = 5,
    VCP_PACKET_DETECTION = 6,
    VCP_PACKET_ERROR = 7,
    VCP_PACKET_COMMAND = 8,
} vcp_packet_type_t;

typedef enum {
    VCP_STATE_READY = 0,
    VCP_STATE_RECEIVING = 1,
    VCP_STATE_PROCESSING = 2,
    VCP_STATE_BUSY = 3,
} vcp_state_code_t;

typedef enum {
    VCP_ACTION_NONE = 0,
    VCP_ACTION_TURN_ON = 1,
    VCP_ACTION_TURN_OFF = 2,
} vcp_action_id_t;

typedef enum {
    VCP_ERROR_NONE = 0,
    VCP_ERROR_BAD_PACKET = 1,
    VCP_ERROR_UNSUPPORTED_FORMAT = 2,
    VCP_ERROR_DETECTOR_NOT_READY = 3,
    VCP_ERROR_BAD_SESSION = 4,
    VCP_ERROR_INTERNAL = 5,
    VCP_ERROR_UNSUPPORTED_ACTION = 6,
} vcp_error_code_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t type;
    uint16_t flags;
    uint32_t length;
} vcp_packet_header_t;

typedef struct __attribute__((packed)) {
    uint32_t protocol_version;
    uint32_t sample_rate;
    uint32_t frame_samples;
    uint16_t bits_per_sample;
    uint16_t channels;
    uint32_t wakeword_count;
    uint32_t detector_ready;
} vcp_hello_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t session_id;
    uint32_t sample_rate;
    uint32_t frame_samples;
    uint16_t bits_per_sample;
    uint16_t channels;
    uint32_t keyword_set_version;
} vcp_session_start_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t session_id;
    uint32_t sequence;
} vcp_audio_frame_prefix_t;

typedef struct __attribute__((packed)) {
    uint32_t session_id;
    uint32_t total_frames;
} vcp_session_end_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t session_id;
    uint32_t state_code;
} vcp_state_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t session_id;
    uint16_t action_id;
    uint16_t keyword_length;
    uint32_t score_milli;
} vcp_detection_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t request_id;
    uint16_t action_id;
    uint16_t keyword_length;
    uint32_t score_milli;
} vcp_command_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t session_id;
    uint16_t error_code;
    uint16_t message_length;
} vcp_error_payload_t;
