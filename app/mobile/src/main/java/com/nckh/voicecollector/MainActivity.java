package com.nckh.voicecollector;

import android.Manifest;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import org.json.JSONException;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 1001;
    private static final String PREFS_NAME = "voice_collector_prefs";
    private static final String PREF_BACKEND_URL = "backend_url";
    private static final String PREF_TRANSPORT_MODE = "transport_mode";
    private static final String PREF_ESP_HOST = "esp_host";
    private static final String PREF_ESP_PORT = "esp_port";

    private static final String MODE_BACKEND = "backend";
    private static final String MODE_ESP_DIRECT = "esp";
    private static final String DEFAULT_BACKEND_URL = "http://10.0.2.2:3000";
    private static final String DEFAULT_ESP_HOST = "192.168.4.1";
    private static final String DEFAULT_ESP_PORT = "3333";
    private static final int KEYWORD_SET_VERSION = 1;
    private static final double ESP_RMS_THRESHOLD = 1100.0;
    private static final int ESP_SILENCE_HANGOVER_FRAMES = 10;

    private final VoiceBackendClient backendClient = new VoiceBackendClient();
    private final ExecutorService ioExecutor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private RadioGroup transportModeRadioGroup;
    private LinearLayout backendSection;
    private LinearLayout espSection;
    private LinearLayout noteSection;
    private LinearLayout sharedRecordingsSection;
    private EditText backendUrlEditText;
    private EditText espHostEditText;
    private EditText espPortEditText;
    private EditText noteEditText;
    private TextView statusTextView;
    private TextView lastDetectionTextView;
    private Button startButton;
    private Button stopButton;
    private Button retryUploadButton;
    private Button refreshButton;
    private RecordingAdapter recordingAdapter;

    private SharedPreferences preferences;
    private MediaRecorder mediaRecorder;
    private File currentRecordingFile;
    private File pendingUploadFile;
    private long recordingStartedAtMs;
    private long pendingDurationMs;
    private volatile boolean shouldRunEspLoop;
    private volatile AudioRecord audioRecord;
    private volatile EspAudioClient activeEspClient;
    private Thread espStreamingThread;
    private boolean isRecording;
    private boolean isUploading;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        preferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        bindViews();
        setupRecyclerView();

        backendUrlEditText.setText(preferences.getString(PREF_BACKEND_URL, DEFAULT_BACKEND_URL));
        espHostEditText.setText(preferences.getString(PREF_ESP_HOST, DEFAULT_ESP_HOST));
        espPortEditText.setText(preferences.getString(PREF_ESP_PORT, DEFAULT_ESP_PORT));
        updateStatus("Idle");
        lastDetectionTextView.setText(getString(R.string.last_detection_empty));
        restoreTransportMode();
        renderModeUi();
        renderButtonState();

        startButton.setOnClickListener(view -> requestPermissionAndStartCapture());
        stopButton.setOnClickListener(view -> {
            if (isEspMode()) {
                stopEspListening();
            } else {
                stopRecordingAndUpload();
            }
        });
        retryUploadButton.setOnClickListener(view -> uploadPendingRecording());
        refreshButton.setOnClickListener(view -> {
            saveTransportSettings();
            if (isEspMode()) {
                checkEspConnection();
            } else {
                refreshRecordings();
            }
        });
        findViewById(R.id.saveBackendButton).setOnClickListener(view -> {
            saveTransportSettings();
            Toast.makeText(this, "Transport settings saved", Toast.LENGTH_SHORT).show();
        });
        transportModeRadioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            preferences.edit()
                    .putString(PREF_TRANSPORT_MODE, checkedId == R.id.espModeRadioButton ? MODE_ESP_DIRECT : MODE_BACKEND)
                    .apply();
            renderModeUi();
        });

        if (!isEspMode()) {
            refreshRecordings();
        }
    }

    private void bindViews() {
        transportModeRadioGroup = findViewById(R.id.transportModeRadioGroup);
        backendSection = findViewById(R.id.backendSection);
        espSection = findViewById(R.id.espSection);
        noteSection = findViewById(R.id.noteSection);
        sharedRecordingsSection = findViewById(R.id.sharedRecordingsSection);
        backendUrlEditText = findViewById(R.id.backendUrlEditText);
        espHostEditText = findViewById(R.id.espHostEditText);
        espPortEditText = findViewById(R.id.espPortEditText);
        noteEditText = findViewById(R.id.noteEditText);
        statusTextView = findViewById(R.id.statusTextView);
        lastDetectionTextView = findViewById(R.id.lastDetectionTextView);
        startButton = findViewById(R.id.startRecordingButton);
        stopButton = findViewById(R.id.stopRecordingButton);
        retryUploadButton = findViewById(R.id.retryUploadButton);
        refreshButton = findViewById(R.id.refreshButton);
    }

    private void setupRecyclerView() {
        RecyclerView recyclerView = findViewById(R.id.recordingsRecyclerView);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recordingAdapter = new RecordingAdapter();
        recyclerView.setAdapter(recordingAdapter);
    }

    private void restoreTransportMode() {
        String savedMode = preferences.getString(PREF_TRANSPORT_MODE, MODE_BACKEND);
        RadioButton radioButton = findViewById(MODE_ESP_DIRECT.equals(savedMode)
                ? R.id.espModeRadioButton
                : R.id.backendModeRadioButton);
        radioButton.setChecked(true);
    }

    private void renderModeUi() {
        boolean espMode = isEspMode();
        backendSection.setVisibility(espMode ? View.GONE : View.VISIBLE);
        espSection.setVisibility(espMode ? View.VISIBLE : View.GONE);
        noteSection.setVisibility(espMode ? View.GONE : View.VISIBLE);
        sharedRecordingsSection.setVisibility(espMode ? View.GONE : View.VISIBLE);
        lastDetectionTextView.setVisibility(espMode ? View.VISIBLE : View.GONE);
        retryUploadButton.setVisibility(espMode ? View.GONE : View.VISIBLE);
        refreshButton.setText(espMode ? R.string.check_esp_connection : R.string.refresh_list);
        startButton.setText(espMode ? R.string.start_listening : R.string.start_recording);
        stopButton.setText(espMode ? R.string.stop_listening : R.string.stop_recording);
        if (espMode) {
            recordingAdapter.submitList(Collections.emptyList());
            updateStatus("ESP direct mode ready");
        } else {
            updateStatus("Backend mode ready");
        }
        renderButtonState();
    }

    private boolean isEspMode() {
        return transportModeRadioGroup.getCheckedRadioButtonId() == R.id.espModeRadioButton;
    }

    private void requestPermissionAndStartCapture() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED) {
            if (isEspMode()) {
                startEspListening();
            } else {
                startRecording();
            }
            return;
        }

        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.RECORD_AUDIO},
                REQUEST_RECORD_AUDIO_PERMISSION
        );
    }

    private void saveTransportSettings() {
        preferences.edit()
                .putString(PREF_BACKEND_URL, backendUrlEditText.getText().toString().trim())
                .putString(PREF_ESP_HOST, espHostEditText.getText().toString().trim())
                .putString(PREF_ESP_PORT, espPortEditText.getText().toString().trim())
                .apply();
    }

    private void startRecording() {
        if (isRecording || isUploading) {
            return;
        }

        File recordingDirectory = new File(getCacheDir(), "recordings");
        if (!recordingDirectory.exists() && !recordingDirectory.mkdirs()) {
            updateStatus("Could not create local recording directory");
            return;
        }

        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        currentRecordingFile = new File(recordingDirectory, "voice_" + timestamp + ".m4a");

        mediaRecorder = new MediaRecorder();
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        mediaRecorder.setAudioSamplingRate(44100);
        mediaRecorder.setAudioEncodingBitRate(128000);
        mediaRecorder.setOutputFile(currentRecordingFile.getAbsolutePath());

        try {
            mediaRecorder.prepare();
            mediaRecorder.start();
            recordingStartedAtMs = System.currentTimeMillis();
            isRecording = true;
            updateStatus("Recording in progress...");
            renderButtonState();
        } catch (IOException | RuntimeException error) {
            releaseRecorder();
            deleteFileQuietly(currentRecordingFile);
            currentRecordingFile = null;
            updateStatus("Could not start recording: " + error.getMessage());
            renderButtonState();
        }
    }

    private void stopRecordingAndUpload() {
        if (!isRecording || mediaRecorder == null) {
            return;
        }

        try {
            mediaRecorder.stop();
            pendingDurationMs = Math.max(System.currentTimeMillis() - recordingStartedAtMs, 0L);
            pendingUploadFile = currentRecordingFile;
            updateStatus("Recording saved locally. Uploading...");
        } catch (RuntimeException error) {
            updateStatus("Recording too short or invalid. Please try again.");
            deleteFileQuietly(currentRecordingFile);
            pendingUploadFile = null;
            pendingDurationMs = 0L;
        } finally {
            releaseRecorder();
            currentRecordingFile = null;
            isRecording = false;
            renderButtonState();
        }

        if (pendingUploadFile != null) {
            uploadPendingRecording();
        }
    }

    private void uploadPendingRecording() {
        if (pendingUploadFile == null || isUploading) {
            return;
        }

        saveTransportSettings();
        final String baseUrl = backendUrlEditText.getText().toString().trim();
        final String note = noteEditText.getText().toString().trim();

        if (baseUrl.isEmpty()) {
            updateStatus("Backend URL is required before upload");
            return;
        }

        isUploading = true;
        renderButtonState();
        updateStatus("Uploading " + pendingUploadFile.getName() + "...");

        ioExecutor.execute(() -> {
            try {
                backendClient.uploadRecording(baseUrl, pendingUploadFile, note, pendingDurationMs);
                deleteFileQuietly(pendingUploadFile);
                pendingUploadFile = null;
                pendingDurationMs = 0L;
                isUploading = false;
                mainHandler.post(() -> {
                    noteEditText.setText("");
                    updateStatus("Upload successful");
                    renderButtonState();
                    refreshRecordings();
                });
            } catch (IOException | JSONException error) {
                isUploading = false;
                mainHandler.post(() -> {
                    updateStatus("Upload failed: " + error.getMessage());
                    renderButtonState();
                });
            }
        });
    }

    private void startEspListening() {
        if (isRecording || isUploading) {
            return;
        }

        saveTransportSettings();
        final String host = espHostEditText.getText().toString().trim();
        final int port = parseEspPort();
        if (host.isEmpty()) {
            updateStatus("ESP host is required");
            return;
        }
        if (port <= 0) {
            updateStatus("ESP port is invalid");
            return;
        }

        isRecording = true;
        shouldRunEspLoop = true;
        renderButtonState();
        updateStatus("Connecting to ESP " + host + ":" + port + "...");

        espStreamingThread = new Thread(() -> runEspListeningLoop(host, port), "esp-stream-thread");
        espStreamingThread.start();
    }

    private void stopEspListening() {
        shouldRunEspLoop = false;
        AudioRecord currentAudioRecord = audioRecord;
        if (currentAudioRecord != null) {
            try {
                currentAudioRecord.stop();
            } catch (IllegalStateException ignored) {
                // The background thread may have already released the recorder.
            }
        }
        EspAudioClient currentClient = activeEspClient;
        if (currentClient != null) {
            try {
                currentClient.close();
            } catch (IOException ignored) {
                // The socket is already closing.
            }
        }
        updateStatus("Stopping ESP listening...");
    }

    private void runEspListeningLoop(String host, int port) {
        EspAudioClient client = new EspAudioClient(host, port);
        activeEspClient = client;

        try {
            EspAudioClient.ServerHello hello = client.connectAndReadHello();
            if (!hello.detectorReady || hello.wakewordCount <= 0) {
                throw new IOException("ESP wakeword engine is not ready. Flash a firmware with models first.");
            }

            postStatus(String.format(
                    Locale.US,
                    "ESP ready: %dHz, frame=%d, models=%d",
                    hello.sampleRate,
                    hello.frameSamples,
                    hello.wakewordCount
            ));

            AudioRecord localAudioRecord = createAudioRecord(hello.sampleRate, hello.frameSamples);
            audioRecord = localAudioRecord;
            localAudioRecord.startRecording();

            short[] frameBuffer = new short[hello.frameSamples];
            boolean sessionActive = false;
            int activeSessionId = 0;
            int sequence = 0;
            int remainingVoiceFrames = 0;

            while (shouldRunEspLoop) {
                int samplesRead = readFullFrame(localAudioRecord, frameBuffer, hello.frameSamples);
                if (samplesRead <= 0) {
                    throw new IOException("Could not read a full frame from microphone");
                }

                boolean voiceDetected = isVoiceDetected(frameBuffer, samplesRead);
                if (voiceDetected && !sessionActive) {
                    activeSessionId = ThreadLocalRandom.current().nextInt(1, Integer.MAX_VALUE);
                    client.sendSessionStart(activeSessionId, hello, KEYWORD_SET_VERSION);
                    sessionActive = true;
                    sequence = 0;
                    remainingVoiceFrames = ESP_SILENCE_HANGOVER_FRAMES;
                    postStatus("Voice segment started, streaming to ESP...");
                }

                if (sessionActive) {
                    client.sendAudioFrame(activeSessionId, sequence++, frameBuffer, samplesRead);
                    consumePollResult(client.drainResponses());

                    if (voiceDetected) {
                        remainingVoiceFrames = ESP_SILENCE_HANGOVER_FRAMES;
                    } else {
                        remainingVoiceFrames--;
                    }

                    if (!voiceDetected && remainingVoiceFrames <= 0) {
                        client.sendSessionEnd(activeSessionId, sequence);
                        consumePollResult(client.drainResponses());
                        sessionActive = false;
                        postStatus("Voice segment ended, waiting for next wakeword...");
                    }
                }
            }

            if (sessionActive) {
                client.sendSessionEnd(activeSessionId, sequence);
                consumePollResult(client.drainResponses());
            }
        } catch (IOException error) {
            postStatus("ESP direct mode failed: " + error.getMessage());
        } finally {
            activeEspClient = null;
            releaseAudioRecord();
            try {
                client.close();
            } catch (IOException ignored) {
                // Already disconnected.
            }
            shouldRunEspLoop = false;
            isRecording = false;
            mainHandler.post(this::renderButtonState);
        }
    }

    private void checkEspConnection() {
        final String host = espHostEditText.getText().toString().trim();
        final int port = parseEspPort();
        if (host.isEmpty() || port <= 0) {
            updateStatus("ESP host/port is invalid");
            return;
        }

        updateStatus("Checking ESP connection...");
        ioExecutor.execute(() -> {
            EspAudioClient client = new EspAudioClient(host, port);
            try {
                EspAudioClient.ServerHello hello = client.connectAndReadHello();
                mainHandler.post(() -> {
                    updateStatus(String.format(
                            Locale.US,
                            "ESP online: %dHz, frame=%d, models=%d, ready=%s",
                            hello.sampleRate,
                            hello.frameSamples,
                            hello.wakewordCount,
                            hello.detectorReady ? "yes" : "no"
                    ));
                    lastDetectionTextView.setText(getString(R.string.last_detection_empty));
                });
            } catch (IOException error) {
                mainHandler.post(() -> updateStatus("Could not reach ESP: " + error.getMessage()));
            } finally {
                try {
                    client.close();
                } catch (IOException ignored) {
                    // Probe connection already closed.
                }
            }
        });
    }

    private void refreshRecordings() {
        final String baseUrl = backendUrlEditText.getText().toString().trim();
        if (baseUrl.isEmpty()) {
            updateStatus("Set backend URL to load shared recordings");
            return;
        }

        updateStatus("Loading recordings...");
        ioExecutor.execute(() -> {
            try {
                List<RecordingItem> recordings = backendClient.fetchRecordings(baseUrl);
                mainHandler.post(() -> {
                    recordingAdapter.submitList(recordings);
                    updateStatus("Loaded " + recordings.size() + " recording(s)");
                });
            } catch (IOException | JSONException error) {
                mainHandler.post(() -> updateStatus("Could not load recordings: " + error.getMessage()));
            }
        });
    }

    private void updateStatus(String message) {
        statusTextView.setText(message);
    }

    private void renderButtonState() {
        startButton.setEnabled(!isRecording && !isUploading);
        stopButton.setEnabled(isRecording);
        retryUploadButton.setEnabled(!isEspMode() && !isRecording && !isUploading && pendingUploadFile != null);
        refreshButton.setEnabled(!isRecording && !isUploading);
        setTransportControlsEnabled(!isRecording && !isUploading);
    }

    private void releaseRecorder() {
        if (mediaRecorder != null) {
            mediaRecorder.reset();
            mediaRecorder.release();
            mediaRecorder = null;
        }
    }

    private void deleteFileQuietly(File file) {
        if (file != null && file.exists()) {
            //noinspection ResultOfMethodCallIgnored
            file.delete();
        }
    }

    private void releaseAudioRecord() {
        AudioRecord currentAudioRecord = audioRecord;
        audioRecord = null;
        if (currentAudioRecord != null) {
            try {
                currentAudioRecord.release();
            } catch (Exception ignored) {
                // Ignore cleanup errors while shutting down the capture pipeline.
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopEspListening();
        releaseRecorder();
        releaseAudioRecord();
        ioExecutor.shutdownNow();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (isEspMode()) {
                    startEspListening();
                } else {
                    startRecording();
                }
            } else {
                updateStatus("Microphone permission denied");
            }
        }
    }

    private int parseEspPort() {
        try {
            return Integer.parseInt(espPortEditText.getText().toString().trim());
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private AudioRecord createAudioRecord(int sampleRate, int frameSamples) throws IOException {
        int minBufferSize = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        if (minBufferSize <= 0) {
            throw new IOException("AudioRecord buffer size invalid for " + sampleRate + "Hz");
        }
        int bufferSize = Math.max(minBufferSize, frameSamples * 8);
        AudioRecord localAudioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
        );
        if (localAudioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            throw new IOException("AudioRecord did not initialize");
        }
        return localAudioRecord;
    }

    private int readFullFrame(AudioRecord localAudioRecord, short[] buffer, int targetSamples) {
        int totalRead = 0;
        while (totalRead < targetSamples && shouldRunEspLoop) {
            int readCount = localAudioRecord.read(buffer, totalRead, targetSamples - totalRead);
            if (readCount <= 0) {
                return readCount;
            }
            totalRead += readCount;
        }
        return totalRead;
    }

    private boolean isVoiceDetected(short[] samples, int sampleCount) {
        double energySum = 0.0;
        for (int index = 0; index < sampleCount; index++) {
            energySum += samples[index] * (double) samples[index];
        }
        double rms = Math.sqrt(energySum / Math.max(sampleCount, 1));
        return rms >= ESP_RMS_THRESHOLD;
    }

    private void consumePollResult(EspAudioClient.PollResult pollResult) {
        if (pollResult.lastState != null) {
            postStatus("ESP state: " + pollResult.lastState.stateLabel);
        }
        if (pollResult.lastError != null) {
            postStatus(pollResult.lastError);
        }
        for (EspAudioClient.Detection detection : pollResult.detections) {
            final String message = String.format(
                    Locale.US,
                    "Detected %s (%s, score=%.3f)",
                    detection.keyword,
                    detection.actionLabel,
                    detection.score
            );
            mainHandler.post(() -> {
                lastDetectionTextView.setText(message);
                updateStatus(message);
            });
        }
    }

    private void postStatus(String message) {
        mainHandler.post(() -> updateStatus(message));
    }

    private void setTransportControlsEnabled(boolean enabled) {
        transportModeRadioGroup.setEnabled(enabled);
        for (int index = 0; index < transportModeRadioGroup.getChildCount(); index++) {
            transportModeRadioGroup.getChildAt(index).setEnabled(enabled);
        }
    }

    public static String formatDuration(long durationMs) {
        long totalSeconds = durationMs / 1000L;
        long minutes = totalSeconds / 60L;
        long seconds = totalSeconds % 60L;
        return String.format(Locale.US, "%02d:%02d", minutes, seconds);
    }

    public static String formatFileSize(long bytes) {
        if (bytes < 1024L) {
            return bytes + " B";
        }
        if (bytes < 1024L * 1024L) {
            return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        }
        return String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0));
    }
}
