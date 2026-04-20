const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");

class RecordingDatabase {
  constructor(databaseFile) {
    fs.mkdirSync(path.dirname(databaseFile), { recursive: true });
    this.db = new DatabaseSync(databaseFile);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        duration_ms INTEGER,
        note TEXT,
        created_at TEXT NOT NULL,
        file_size_bytes INTEGER NOT NULL,
        storage_path TEXT NOT NULL
      )
    `);
    this.insertStatement = this.db.prepare(`
      INSERT INTO recordings (
        filename,
        original_filename,
        mime_type,
        duration_ms,
        note,
        created_at,
        file_size_bytes,
        storage_path
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    this.listStatement = this.db.prepare(`
      SELECT
        id,
        filename,
        original_filename,
        mime_type,
        duration_ms,
        note,
        created_at,
        file_size_bytes,
        storage_path
      FROM recordings
      ORDER BY datetime(created_at) DESC, id DESC
    `);
    this.getByIdStatement = this.db.prepare(`
      SELECT
        id,
        filename,
        original_filename,
        mime_type,
        duration_ms,
        note,
        created_at,
        file_size_bytes,
        storage_path
      FROM recordings
      WHERE id = ?
    `);
  }

  create(recording) {
    const result = this.insertStatement.run(
      recording.filename,
      recording.originalFilename,
      recording.mimeType,
      recording.durationMs,
      recording.note,
      recording.createdAt,
      recording.fileSizeBytes,
      recording.storagePath
    );

    return this.getById(Number(result.lastInsertRowid));
  }

  list() {
    return this.listStatement.all();
  }

  getById(id) {
    return this.getByIdStatement.get(id) || null;
  }

  close() {
    this.db.close();
  }
}

module.exports = {
  RecordingDatabase,
};
