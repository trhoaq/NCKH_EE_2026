# ESP32-S3 Command Receiver Firmware

ESP-IDF firmware that:

- exposes a SoftAP for direct phone-to-device communication
- accepts resolved action commands over raw TCP
- validates supported `action_id` values from the phone
- maps commands to logical `TURN_ON` / `TURN_OFF` actions

## Layout

- `main/`: ESP-IDF app and transport server

## Active behavior

- The phone performs speech recognition / command inference.
- The phone sends a fixed `action_id` to the ESP.
- The ESP applies the action and returns an acknowledgement over the same TCP protocol.
- GPIO 48 is driven HIGH for `TURN_ON` and LOW for `TURN_OFF`.
