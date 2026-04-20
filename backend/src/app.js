const express = require("express");
const fs = require("fs");
const path = require("path");
const { RecordingDatabase } = require("./database");
const { buildUploader } = require("./storage");

function parseDurationMs(rawValue) {
  if (rawValue === undefined || rawValue === null || rawValue === "") {
    return null;
  }

  const value = Number.parseInt(rawValue, 10);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function toRecordingResponse(recording, request) {
  return {
    id: recording.id,
    filename: recording.filename,
    originalFilename: recording.original_filename,
    mimeType: recording.mime_type,
    durationMs: recording.duration_ms,
    note: recording.note,
    createdAt: recording.created_at,
    fileSizeBytes: recording.file_size_bytes,
    fileUrl: `${request.protocol}://${request.get("host")}/recordings/${recording.id}/file`,
  };
}

function createApp(options) {
  const storageRoot = options.storageRoot;
  const databaseFile = options.databaseFile;

  fs.mkdirSync(storageRoot, { recursive: true });
  fs.mkdirSync(path.dirname(databaseFile), { recursive: true });

  const database = new RecordingDatabase(databaseFile);
  const uploader = buildUploader(storageRoot);
  const app = express();

  app.use(express.json());

  app.get("/health", (req, res) => {
    res.json({
      ok: true,
      storageRoot,
      databaseFile,
      recordingCount: database.list().length,
    });
  });

  app.get("/recordings", (req, res) => {
    const recordings = database.list().map((recording) => toRecordingResponse(recording, req));
    res.json({
      recordings,
    });
  });

  app.post("/recordings", uploader.single("audio"), (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        error: "audio file is required",
      });
    }

    const now = new Date().toISOString();
    const note = typeof req.body.note === "string" ? req.body.note.trim() : "";
    const durationMs = parseDurationMs(req.body.duration_ms);

    const created = database.create({
      filename: req.savedFilename,
      originalFilename: req.file.originalname,
      mimeType: req.file.mimetype || "application/octet-stream",
      durationMs,
      note: note.length > 0 ? note : null,
      createdAt: now,
      fileSizeBytes: req.file.size,
      storagePath: req.savedRelativePath,
    });

    return res.status(201).json({
      recording: toRecordingResponse(created, req),
    });
  });

  app.get("/recordings/:id/file", (req, res) => {
    const id = Number.parseInt(req.params.id, 10);
    const recording = database.getById(id);

    if (!recording) {
      return res.status(404).json({
        error: "recording not found",
      });
    }

    const absolutePath = path.join(storageRoot, recording.storage_path);
    if (!fs.existsSync(absolutePath)) {
      return res.status(404).json({
        error: "audio file not found on disk",
      });
    }

    res.type(recording.mime_type);
    return res.sendFile(absolutePath);
  });

  app.use((err, req, res, next) => {
    if (err && err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({
        error: "audio file exceeds 50MB limit",
      });
    }

    if (err) {
      return res.status(500).json({
        error: err.message || "unexpected server error",
      });
    }

    return next();
  });

  return {
    app,
    database,
  };
}

module.exports = {
  createApp,
};
