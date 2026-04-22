package com.nckh.voicecollector;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import java.io.IOException;
import java.util.ArrayDeque;

public final class StreamingCommandRecognizer {
    public interface Listener {
        void onCommandDetected(TfliteCommandClassifier.Prediction prediction);
        void onStatus(String message);
        void onError(String message);
    }

    private final Context context;
    private final String modelAssetName;
    private final String metadataAssetName;
    private final Listener listener;

    private volatile boolean running;
    private Thread workerThread;
    private AudioRecord audioRecord;
    private TfliteCommandClassifier classifier;
    private long lastTriggerAtMs;
    private final ArrayDeque<float[]> probabilityHistory = new ArrayDeque<>();

    public StreamingCommandRecognizer(Context context, String modelAssetName, String metadataAssetName, Listener listener) {
        this.context = context.getApplicationContext();
        this.modelAssetName = modelAssetName;
        this.metadataAssetName = metadataAssetName;
        this.listener = listener;
    }

    public void start() throws IOException {
        if (running) {
            return;
        }
        classifier = TfliteCommandClassifier.fromAssets(context, modelAssetName, metadataAssetName);
        int sampleRate = classifier.getSampleRate();
        int minBufferSize = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        if (minBufferSize <= 0) {
            throw new IOException("AudioRecord buffer size is invalid");
        }

        int bufferSize = Math.max(minBufferSize, classifier.getWindowSamples() * 2);
        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
        );
        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            releaseAudioRecord();
            throw new IOException("AudioRecord did not initialize");
        }

        running = true;
        probabilityHistory.clear();
        lastTriggerAtMs = 0L;
        workerThread = new Thread(this::runLoop, "command-recognizer-thread");
        workerThread.start();
    }

    public void stop() {
        running = false;
        if (audioRecord != null) {
            try {
                audioRecord.stop();
            } catch (IllegalStateException ignored) {
                // Already stopping.
            }
        }
        if (workerThread != null) {
            try {
                workerThread.join(500L);
            } catch (InterruptedException interruptedException) {
                Thread.currentThread().interrupt();
            }
            workerThread = null;
        }
        releaseAudioRecord();
        if (classifier != null) {
            classifier.close();
            classifier = null;
        }
        probabilityHistory.clear();
    }

    private void runLoop() {
        short[] captureChunk = new short[Math.max(160, classifier.getStreamHopSamples() / 4)];
        short[] ringBuffer = new short[classifier.getWindowSamples()];
        int writePosition = 0;
        int totalSamples = 0;
        int samplesSinceLastInference = 0;

        try {
            audioRecord.startRecording();
            listener.onStatus("DL recognizer listening...");

            while (running) {
                int samplesRead = audioRecord.read(captureChunk, 0, captureChunk.length);
                if (samplesRead <= 0) {
                    listener.onError("Audio capture failed");
                    break;
                }

                for (int index = 0; index < samplesRead; index++) {
                    ringBuffer[writePosition] = captureChunk[index];
                    writePosition = (writePosition + 1) % ringBuffer.length;
                }

                totalSamples = Math.min(ringBuffer.length, totalSamples + samplesRead);
                samplesSinceLastInference += samplesRead;
                if (totalSamples < ringBuffer.length || samplesSinceLastInference < classifier.getStreamHopSamples()) {
                    continue;
                }
                samplesSinceLastInference = 0;

                short[] window = extractWindow(ringBuffer, writePosition);
                TfliteCommandClassifier.Prediction prediction = classifier.predict(window, classifier.getSampleRate());
                if (prediction.probabilities != null) {
                    probabilityHistory.addLast(prediction.probabilities.clone());
                    while (probabilityHistory.size() > classifier.getSmoothingWindows()) {
                        probabilityHistory.removeFirst();
                    }
                }

                TfliteCommandClassifier.Prediction smoothed = smoothPrediction(prediction);
                if (smoothed.actionId == 0) {
                    continue;
                }
                long now = System.currentTimeMillis();
                if (smoothed.confidence < classifier.getThreshold()) {
                    continue;
                }
                if (now - lastTriggerAtMs < classifier.getCooldownMs()) {
                    continue;
                }

                lastTriggerAtMs = now;
                listener.onCommandDetected(smoothed);
            }
        } catch (RuntimeException error) {
            listener.onError("Recognizer runtime failed: " + error.getMessage());
        } finally {
            running = false;
            releaseAudioRecord();
        }
    }

    private TfliteCommandClassifier.Prediction smoothPrediction(TfliteCommandClassifier.Prediction fallback) {
        if (probabilityHistory.isEmpty()) {
            return fallback;
        }

        int classCount = probabilityHistory.peekFirst().length;
        float[] averaged = new float[classCount];
        for (float[] probabilities : probabilityHistory) {
            for (int index = 0; index < classCount; index++) {
                averaged[index] += probabilities[index];
            }
        }
        for (int index = 0; index < classCount; index++) {
            averaged[index] /= probabilityHistory.size();
        }

        int bestIndex = 0;
        for (int index = 1; index < averaged.length; index++) {
            if (averaged[index] > averaged[bestIndex]) {
                bestIndex = index;
            }
        }
        String label;
        int actionId;
        if (bestIndex == 0) {
            label = "on";
            actionId = 1;
        } else if (bestIndex == 1) {
            label = "off";
            actionId = 2;
        } else if (bestIndex == 2) {
            label = "unknown";
            actionId = 0;
        } else {
            label = "silence";
            actionId = 0;
        }
        return new TfliteCommandClassifier.Prediction(actionId, label, averaged[bestIndex], averaged);
    }

    private short[] extractWindow(short[] ringBuffer, int writePosition) {
        short[] output = new short[ringBuffer.length];
        int tail = ringBuffer.length - writePosition;
        System.arraycopy(ringBuffer, writePosition, output, 0, tail);
        System.arraycopy(ringBuffer, 0, output, tail, writePosition);
        return output;
    }

    private void releaseAudioRecord() {
        if (audioRecord != null) {
            try {
                audioRecord.release();
            } catch (Exception ignored) {
                // Ignore cleanup failures.
            }
            audioRecord = null;
        }
    }
}
