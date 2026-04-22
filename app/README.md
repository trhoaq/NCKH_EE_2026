# Voice Collector Android App

Native Android Java client with two transport modes:

- `Backend`: record to file and upload to the local Node.js backend.
- `ESP Direct`: recognize a spoken command on the phone, map it to a fixed action, and send that action directly to an ESP32-S3 over raw TCP.

## Expected workflow

1. Start the backend from `../backend`.
2. Open this Java Android project in Android Studio.
3. Set the backend URL:
   - Emulator: `http://10.0.2.2:3000`
   - Physical device: `http://<your-lan-ip>:3000`
4. Choose the transport mode you want:
   - `Backend` mode: save the backend URL, record audio, then stop to upload.
   - `ESP Direct` mode: set the ESP host/port, tap `Check ESP`, then tap `Start command listen`.
5. Use `Refresh list` to load the shared recordings list from the backend when backend mode is active.

## Notes

- Backend mode stores recordings in app cache before upload.
- Failed backend uploads stay local and can be retried manually with `Retry last upload`.
- ESP direct mode uses the local DL pipeline on the phone and sends only a resolved `action_id` to the ESP.
- The ESP no longer runs wakeword or command inference; it only validates incoming commands and applies control logic.
- The active command model pair is:
  - `app/mobile/src/main/assets/command_model.tflite`
  - `app/mobile/src/main/assets/command_model_meta.json`
