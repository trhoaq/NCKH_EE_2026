package com.nckh.voicecollector;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class VoiceBackendClient {
    private static final int CONNECT_TIMEOUT_MS = 15000;
    private static final int READ_TIMEOUT_MS = 20000;

    public List<RecordingItem> fetchRecordings(String baseUrl) throws IOException, JSONException {
        HttpURLConnection connection = openConnection(buildEndpoint(baseUrl, "/recordings"), "GET");
        try {
            int responseCode = connection.getResponseCode();
            String responseBody = readResponseBody(connection, responseCode);
            if (responseCode < 200 || responseCode >= 300) {
                throw new IOException(extractErrorMessage(responseBody, responseCode));
            }

            JSONObject jsonObject = new JSONObject(responseBody);
            JSONArray recordingsJson = jsonObject.getJSONArray("recordings");
            List<RecordingItem> recordings = new ArrayList<>();
            for (int i = 0; i < recordingsJson.length(); i++) {
                recordings.add(RecordingItem.fromJson(recordingsJson.getJSONObject(i)));
            }
            return recordings;
        } finally {
            connection.disconnect();
        }
    }

    public RecordingItem uploadRecording(String baseUrl, File audioFile, String note, long durationMs)
            throws IOException, JSONException {
        List<String> mimeTypesToTry = buildMimeTypeCandidates(audioFile);
        IOException lastError = null;

        for (int index = 0; index < mimeTypesToTry.size(); index++) {
            String mimeType = mimeTypesToTry.get(index);
            try {
                return uploadRecordingWithMimeType(baseUrl, audioFile, note, durationMs, mimeType);
            } catch (HttpStatusException error) {
                if (error.statusCode != 415 || index == mimeTypesToTry.size() - 1) {
                    throw error;
                }
                lastError = error;
            }
        }

        if (lastError != null) {
            throw lastError;
        }

        throw new IOException("Upload failed before sending request");
    }

    private HttpURLConnection openConnection(String targetUrl, String method) throws IOException {
        URL url = new URL(targetUrl);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod(method);
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        connection.setRequestProperty("Accept", "application/json");
        return connection;
    }

    private String buildEndpoint(String baseUrl, String endpointPath) {
        String normalizedBaseUrl = baseUrl;
        if (normalizedBaseUrl.endsWith("/")) {
            normalizedBaseUrl = normalizedBaseUrl.substring(0, normalizedBaseUrl.length() - 1);
        }
        return normalizedBaseUrl + endpointPath;
    }

    private void writeFormField(DataOutputStream outputStream, String boundary, String name, String value)
            throws IOException {
        outputStream.writeBytes("--" + boundary + "\r\n");
        outputStream.writeBytes("Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n");
        outputStream.write(value.getBytes(StandardCharsets.UTF_8));
        outputStream.writeBytes("\r\n");
    }

    private void writeFileField(
            DataOutputStream outputStream,
            String boundary,
            String fieldName,
            File file,
            String mimeType
    )
            throws IOException {
        outputStream.writeBytes("--" + boundary + "\r\n");
        outputStream.writeBytes(
                "Content-Disposition: form-data; name=\"" + fieldName + "\"; filename=\"" + file.getName() + "\"\r\n"
        );
        outputStream.writeBytes("Content-Type: " + mimeType + "\r\n");
        outputStream.writeBytes("Content-Transfer-Encoding: binary\r\n\r\n");

        try (BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }

        outputStream.writeBytes("\r\n");
    }

    private String readResponseBody(HttpURLConnection connection, int responseCode) throws IOException {
        if (responseCode >= 400 && connection.getErrorStream() == null) {
            return "";
        }

        try (BufferedInputStream inputStream = new BufferedInputStream(
                responseCode >= 400 ? connection.getErrorStream() : connection.getInputStream());
             ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            return outputStream.toString(StandardCharsets.UTF_8.name());
        }
    }

    private String extractErrorMessage(String responseBody, int responseCode) {
        if (responseBody == null || responseBody.isEmpty()) {
            return "Request failed with HTTP " + responseCode;
        }

        try {
            JSONObject jsonObject = new JSONObject(responseBody);
            String errorMessage = jsonObject.optString("error");
            if (errorMessage != null && !errorMessage.isEmpty()) {
                return errorMessage;
            }
        } catch (JSONException ignored) {
            // Return raw body when response is not JSON.
        }

        return responseBody;
    }

    private RecordingItem uploadRecordingWithMimeType(
            String baseUrl,
            File audioFile,
            String note,
            long durationMs,
            String mimeType
    ) throws IOException, JSONException {
        String boundary = "Boundary-" + UUID.randomUUID();
        HttpURLConnection connection = openConnection(buildEndpoint(baseUrl, "/recordings"), "POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

        try {
            try (DataOutputStream outputStream = new DataOutputStream(
                    new BufferedOutputStream(connection.getOutputStream()))) {
                writeFormField(outputStream, boundary, "note", note == null ? "" : note);
                writeFormField(outputStream, boundary, "duration_ms", String.valueOf(durationMs));
                writeFileField(outputStream, boundary, "audio", audioFile, mimeType);
                outputStream.writeBytes("--" + boundary + "--\r\n");
                outputStream.flush();
            }

            int responseCode = connection.getResponseCode();
            String responseBody = readResponseBody(connection, responseCode);
            if (responseCode < 200 || responseCode >= 300) {
                throw new HttpStatusException(responseCode, extractErrorMessage(responseBody, responseCode));
            }

            JSONObject payload = new JSONObject(responseBody);
            return RecordingItem.fromJson(payload.getJSONObject("recording"));
        } finally {
            connection.disconnect();
        }
    }

    private List<String> buildMimeTypeCandidates(File audioFile) {
        String fileName = audioFile.getName().toLowerCase(Locale.US);
        if (fileName.endsWith(".m4a")) {
            return Arrays.asList("audio/mp4", "audio/x-m4a", "audio/m4a", "application/octet-stream");
        }
        if (fileName.endsWith(".wav")) {
            return Arrays.asList("audio/wav", "audio/x-wav", "application/octet-stream");
        }
        if (fileName.endsWith(".mp3")) {
            return Arrays.asList("audio/mpeg", "application/octet-stream");
        }
        return Arrays.asList("application/octet-stream");
    }

    private static final class HttpStatusException extends IOException {
        final int statusCode;

        HttpStatusException(int statusCode, String message) {
            super(message);
            this.statusCode = statusCode;
        }
    }
}
