package com.nckh.voicecollector;

import org.json.JSONObject;

public class RecordingItem {
    private final int id;
    private final String note;
    private final String createdAt;
    private final long durationMs;
    private final long fileSizeBytes;
    private final String fileUrl;
    private final String originalFilename;
    private final String mimeType;

    public RecordingItem(
            int id,
            String note,
            String createdAt,
            long durationMs,
            long fileSizeBytes,
            String fileUrl,
            String originalFilename,
            String mimeType
    ) {
        this.id = id;
        this.note = note;
        this.createdAt = createdAt;
        this.durationMs = durationMs;
        this.fileSizeBytes = fileSizeBytes;
        this.fileUrl = fileUrl;
        this.originalFilename = originalFilename;
        this.mimeType = mimeType;
    }

    public static RecordingItem fromJson(JSONObject jsonObject) {
        return new RecordingItem(
                jsonObject.optInt("id"),
                jsonObject.optString("note", ""),
                jsonObject.optString("createdAt", ""),
                jsonObject.optLong("durationMs", 0L),
                jsonObject.optLong("fileSizeBytes", 0L),
                jsonObject.optString("fileUrl", ""),
                jsonObject.optString("originalFilename", ""),
                jsonObject.optString("mimeType", "")
        );
    }

    public int getId() {
        return id;
    }

    public String getNote() {
        return note;
    }

    public String getCreatedAt() {
        return createdAt;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public long getFileSizeBytes() {
        return fileSizeBytes;
    }

    public String getFileUrl() {
        return fileUrl;
    }

    public String getOriginalFilename() {
        return originalFilename;
    }

    public String getMimeType() {
        return mimeType;
    }
}
