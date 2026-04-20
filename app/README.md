# Voice Collector Android App

Native Android Java client with two transport modes:

- `Backend`: record to file and upload to the local Node.js backend.
- `ESP Direct`: capture PCM audio frames and stream short wakeword segments directly to an ESP32-S3 over raw TCP.

## Expected workflow

1. Start the backend from `../backend`.
2. Open this Java Android project in Android Studio.
3. Set the backend URL:
   - Emulator: `http://10.0.2.2:3000`
   - Physical device: `http://<your-lan-ip>:3000`
4. Choose the transport mode you want:
   - `Backend` mode: save the backend URL, record audio, then stop to upload.
   - `ESP Direct` mode: set the ESP host/port, tap `Check ESP`, then tap `Start listening`.
5. Use `Refresh list` to load the shared recordings list from the backend when backend mode is active.

## Notes

- Backend mode stores recordings in app cache before upload.
- Failed backend uploads stay local and can be retried manually with `Retry last upload`.
- ESP direct mode uses `AudioRecord`, simple RMS-based segmentation, and the binary protocol implemented by the firmware under `../firmware`.
- In ESP direct mode, the app sends one short buffered voice segment at a time; the ESP runs `lightwake` detection when the segment ends.
