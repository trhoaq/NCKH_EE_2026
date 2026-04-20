# ESP32-S3 Wakeword Bridge Firmware

ESP-IDF firmware that:

- exposes a SoftAP for direct phone-to-device communication
- accepts PCM audio frames over raw TCP
- buffers each short captured session and routes it into a Rust static library backed by `model/lightwake`
- maps wakeword detections to logical `TURN_ON` / `TURN_OFF` actions

## Layout

- `main/`: ESP-IDF app and transport server
- `components/rust_bridge/`: C shim + Cargo-backed Rust static library integration
- `../model/`: lightwake manifest and `.lww` model asset
