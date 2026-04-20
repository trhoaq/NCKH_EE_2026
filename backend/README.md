# Voice Collector Backend

Node.js backend for the Android voice collector MVP.

## Run

```bash
npm install
npm start
```

Server defaults:

- Port: `3000`
- Audio storage: `backend/storage/audio`
- SQLite database: `backend/storage/recordings.sqlite`

## Environment variables

- `PORT`
- `VOICE_STORAGE_DIR`
- `VOICE_DATABASE_FILE`
