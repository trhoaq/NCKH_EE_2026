package com.nckh.voicecollector;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

public final class TfliteCommandClassifier implements AutoCloseable {
    private static final int ACTION_NONE = 0;
    private static final int ACTION_TURN_ON = 1;
    private static final int ACTION_TURN_OFF = 2;

    private final int sampleRate;
    private final int windowSamples;
    private final int frameSize;
    private final int frameHop;
    private final int fftSize;
    private final int melBins;
    private final int targetFrames;
    private final int streamHopSamples;
    private final float threshold;
    private final int smoothingWindows;
    private final int cooldownMs;
    private final String[] labels;
    private final float[] featureMean;
    private final float[] featureStd;
    private final float[][] melFilterBank;
    private final float[] analysisWindow;
    private final Interpreter interpreter;

    private TfliteCommandClassifier(
            int sampleRate,
            int windowSamples,
            int frameSize,
            int frameHop,
            int fftSize,
            int melBins,
            int targetFrames,
            int streamHopSamples,
            float threshold,
            int smoothingWindows,
            int cooldownMs,
            String[] labels,
            float[] featureMean,
            float[] featureStd,
            Interpreter interpreter
    ) {
        this.sampleRate = sampleRate;
        this.windowSamples = windowSamples;
        this.frameSize = frameSize;
        this.frameHop = frameHop;
        this.fftSize = fftSize;
        this.melBins = melBins;
        this.targetFrames = targetFrames;
        this.streamHopSamples = streamHopSamples;
        this.threshold = threshold;
        this.smoothingWindows = smoothingWindows;
        this.cooldownMs = cooldownMs;
        this.labels = labels;
        this.featureMean = featureMean;
        this.featureStd = featureStd;
        this.interpreter = interpreter;
        this.melFilterBank = buildMelFilterBank(sampleRate, fftSize, melBins);
        this.analysisWindow = buildHannWindow(frameSize);
    }

    public static TfliteCommandClassifier fromAssets(
            Context context,
            String modelAssetName,
            String metadataAssetName
    ) throws IOException {
        String metadataPayload = readAssetText(context, metadataAssetName);
        JSONObject root;
        try {
            root = new JSONObject(metadataPayload);
        } catch (JSONException error) {
            throw new IOException("Could not parse model metadata JSON", error);
        }
        MappedByteBuffer modelBuffer = loadModelFile(context, modelAssetName);
        Interpreter interpreter = new Interpreter(modelBuffer);

        try {
            return new TfliteCommandClassifier(
                    root.getInt("sample_rate"),
                    root.getInt("window_samples"),
                    root.getInt("frame_size"),
                    root.getInt("frame_hop"),
                    root.getInt("fft_size"),
                    root.getInt("mel_bins"),
                    root.getInt("target_frames"),
                    root.optInt("stream_hop_samples", root.getInt("frame_hop") * 25),
                    (float) root.getDouble("threshold"),
                    root.getInt("smoothing_windows"),
                    root.getInt("cooldown_ms"),
                    readStringArray(root.getJSONArray("labels")),
                    readFloatArray(root.getJSONArray("feature_mean")),
                    readFloatArray(root.getJSONArray("feature_std")),
                    interpreter
            );
        } catch (JSONException error) {
            interpreter.close();
            throw new IOException("Could not read model metadata fields", error);
        }
    }

    public Prediction predict(short[] pcmSamples, int pcmSampleRate) {
        float[] samples = shortsToFloats(pcmSamples);
        return predict(samples, pcmSampleRate);
    }

    public Prediction predict(float[] rawSamples, int rawSampleRate) {
        float[] samples = resampleLinear(rawSamples, rawSampleRate, sampleRate);
        samples = normalizeAmplitude(samples);
        samples = trimSilence(samples);
        samples = fitToWindow(samples);

        float[][] logMel = computeLogMel(samples);
        normalizeLogMel(logMel);

        float[][][][] input = new float[1][targetFrames][melBins][1];
        for (int frameIndex = 0; frameIndex < targetFrames; frameIndex++) {
            for (int melIndex = 0; melIndex < melBins; melIndex++) {
                input[0][frameIndex][melIndex][0] = logMel[frameIndex][melIndex];
            }
        }

        float[][] output = new float[1][labels.length];
        interpreter.run(input, output);
        float[] probabilities = output[0];

        int bestIndex = 0;
        for (int index = 1; index < probabilities.length; index++) {
            if (probabilities[index] > probabilities[bestIndex]) {
                bestIndex = index;
            }
        }

        String label = labels[bestIndex];
        int actionId;
        if ("on".equals(label)) {
            actionId = ACTION_TURN_ON;
        } else if ("off".equals(label)) {
            actionId = ACTION_TURN_OFF;
        } else {
            actionId = ACTION_NONE;
        }
        return new Prediction(actionId, label, probabilities[bestIndex], probabilities);
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public int getWindowSamples() {
        return windowSamples;
    }

    public int getStreamHopSamples() {
        return streamHopSamples;
    }

    public float getThreshold() {
        return threshold;
    }

    public int getSmoothingWindows() {
        return smoothingWindows;
    }

    public int getCooldownMs() {
        return cooldownMs;
    }

    @Override
    public void close() {
        interpreter.close();
    }

    private float[][] computeLogMel(float[] samples) {
        int frameCount = 1 + Math.max(0, (samples.length - frameSize) / frameHop);
        float[][] frames = new float[frameCount][melBins];
        float[] real = new float[fftSize];
        float[] imag = new float[fftSize];
        float[] power = new float[(fftSize / 2) + 1];

        for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
            int start = frameIndex * frameHop;
            for (int index = 0; index < fftSize; index++) {
                real[index] = 0f;
                imag[index] = 0f;
            }
            for (int index = 0; index < frameSize; index++) {
                real[index] = samples[start + index] * analysisWindow[index];
            }
            fft(real, imag);
            for (int bin = 0; bin < power.length; bin++) {
                power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
            }
            for (int melIndex = 0; melIndex < melBins; melIndex++) {
                float energy = 0f;
                for (int bin = 0; bin < power.length; bin++) {
                    energy += melFilterBank[melIndex][bin] * power[bin];
                }
                frames[frameIndex][melIndex] = (float) Math.log(energy + 1e-5f);
            }
        }

        if (frameCount == targetFrames) {
            return frames;
        }

        float[][] resized = new float[targetFrames][melBins];
        if (frameCount == 1) {
            for (int frameIndex = 0; frameIndex < targetFrames; frameIndex++) {
                System.arraycopy(frames[0], 0, resized[frameIndex], 0, melBins);
            }
            return resized;
        }

        for (int targetIndex = 0; targetIndex < targetFrames; targetIndex++) {
            float sourcePosition = targetFrames == 1
                    ? 0f
                    : targetIndex * (frameCount - 1f) / (targetFrames - 1f);
            int left = (int) Math.floor(sourcePosition);
            int right = Math.min(left + 1, frameCount - 1);
            float alpha = sourcePosition - left;
            for (int melIndex = 0; melIndex < melBins; melIndex++) {
                resized[targetIndex][melIndex] =
                        frames[left][melIndex] * (1f - alpha) + frames[right][melIndex] * alpha;
            }
        }
        return resized;
    }

    private void normalizeLogMel(float[][] logMel) {
        for (int frameIndex = 0; frameIndex < logMel.length; frameIndex++) {
            for (int melIndex = 0; melIndex < melBins; melIndex++) {
                logMel[frameIndex][melIndex] =
                        (logMel[frameIndex][melIndex] - featureMean[melIndex]) / featureStd[melIndex];
            }
        }
    }

    private float[] fitToWindow(float[] samples) {
        if (samples.length >= windowSamples) {
            int center = samples.length / 2;
            int half = windowSamples / 2;
            int start = Math.max(0, center - half);
            int end = Math.min(samples.length, start + windowSamples);
            start = end - windowSamples;
            float[] output = new float[windowSamples];
            System.arraycopy(samples, start, output, 0, windowSamples);
            return output;
        }
        float[] output = new float[windowSamples];
        int offset = (windowSamples - samples.length) / 2;
        System.arraycopy(samples, 0, output, offset, samples.length);
        return output;
    }

    private float[] trimSilence(float[] samples) {
        if (samples.length <= frameSize) {
            return samples;
        }
        int frameCount = 1 + Math.max(0, (samples.length - frameSize) / frameHop);
        float[] rms = new float[frameCount];
        float maxRms = 0f;
        for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
            int start = frameIndex * frameHop;
            float energy = 0f;
            for (int index = 0; index < frameSize; index++) {
                float sample = samples[start + index];
                energy += sample * sample;
            }
            rms[frameIndex] = (float) Math.sqrt(energy / frameSize + 1e-8f);
            if (rms[frameIndex] > maxRms) {
                maxRms = rms[frameIndex];
            }
        }

        float threshold = Math.max(maxRms * 0.12f, 0.01f);
        int firstActive = -1;
        int lastActive = -1;
        for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
            if (rms[frameIndex] >= threshold) {
                if (firstActive < 0) {
                    firstActive = frameIndex;
                }
                lastActive = frameIndex;
            }
        }
        if (firstActive < 0) {
            return samples;
        }

        int paddedStartFrame = Math.max(0, firstActive - 2);
        int paddedEndFrame = Math.min(frameCount - 1, lastActive + 2);
        int start = paddedStartFrame * frameHop;
        int end = Math.min(samples.length, paddedEndFrame * frameHop + frameSize);
        int length = Math.max(1, end - start);
        float[] trimmed = new float[length];
        System.arraycopy(samples, start, trimmed, 0, length);
        return trimmed;
    }

    private static float[] normalizeAmplitude(float[] samples) {
        float peak = 0f;
        for (float sample : samples) {
            peak = Math.max(peak, Math.abs(sample));
        }
        if (peak < 1e-5f) {
            return samples;
        }
        float[] normalized = new float[samples.length];
        for (int index = 0; index < samples.length; index++) {
            normalized[index] = samples[index] / peak;
        }
        return normalized;
    }

    private static float[] shortsToFloats(short[] samples) {
        float[] output = new float[samples.length];
        for (int index = 0; index < samples.length; index++) {
            output[index] = samples[index] / 32768f;
        }
        return output;
    }

    private static float[] resampleLinear(float[] samples, int inputRate, int outputRate) {
        if (samples.length == 0 || inputRate == outputRate) {
            return samples;
        }
        int outputLength = Math.max(1, Math.round(samples.length * (outputRate / (float) inputRate)));
        float[] output = new float[outputLength];
        for (int index = 0; index < outputLength; index++) {
            float sourcePosition = index / (outputRate / (float) inputRate);
            int left = (int) Math.floor(sourcePosition);
            int right = Math.min(left + 1, samples.length - 1);
            float alpha = sourcePosition - left;
            output[index] = samples[left] * (1f - alpha) + samples[right] * alpha;
        }
        return output;
    }

    private static float[] buildHannWindow(int size) {
        float[] window = new float[size];
        if (size <= 1) {
            return window;
        }
        for (int index = 0; index < size; index++) {
            window[index] = (float) (0.5 - 0.5 * Math.cos((2.0 * Math.PI * index) / (size - 1)));
        }
        return window;
    }

    private static float[][] buildMelFilterBank(int sampleRate, int fftSize, int melBins) {
        float minMel = hzToMel(20f);
        float maxMel = hzToMel(sampleRate / 2f);
        float[] melPoints = new float[melBins + 2];
        for (int index = 0; index < melPoints.length; index++) {
            melPoints[index] = minMel + (maxMel - minMel) * index / (melBins + 1f);
        }
        float[] hzPoints = new float[melPoints.length];
        int[] bins = new int[melPoints.length];
        int maxBin = fftSize / 2;
        for (int index = 0; index < melPoints.length; index++) {
            hzPoints[index] = melToHz(melPoints[index]);
            bins[index] = Math.min(maxBin, (int) Math.floor((maxBin + 1f) * hzPoints[index] / (sampleRate / 2f)));
        }

        float[][] bank = new float[melBins][maxBin + 1];
        for (int melIndex = 0; melIndex < melBins; melIndex++) {
            int left = bins[melIndex];
            int center = Math.max(left + 1, bins[melIndex + 1]);
            int right = Math.max(center + 1, bins[melIndex + 2]);
            for (int bin = left; bin < Math.min(center, bank[melIndex].length); bin++) {
                bank[melIndex][bin] = (bin - left) / (float) Math.max(1, center - left);
            }
            for (int bin = center; bin < Math.min(right, bank[melIndex].length); bin++) {
                bank[melIndex][bin] = (right - bin) / (float) Math.max(1, right - center);
            }
        }
        return bank;
    }

    private static float hzToMel(float hz) {
        return (float) (2595.0 * Math.log10(1.0 + hz / 700.0));
    }

    private static float melToHz(float mel) {
        return (float) (700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0));
    }

    private static void fft(float[] real, float[] imag) {
        int n = real.length;
        int levels = 31 - Integer.numberOfLeadingZeros(n);
        for (int index = 0; index < n; index++) {
            int reversed = Integer.reverse(index) >>> (32 - levels);
            if (reversed > index) {
                float tmpReal = real[index];
                real[index] = real[reversed];
                real[reversed] = tmpReal;
                float tmpImag = imag[index];
                imag[index] = imag[reversed];
                imag[reversed] = tmpImag;
            }
        }

        for (int size = 2; size <= n; size <<= 1) {
            int halfSize = size / 2;
            float angleStep = (float) (-2.0 * Math.PI / size);
            for (int start = 0; start < n; start += size) {
                for (int offset = 0; offset < halfSize; offset++) {
                    float angle = angleStep * offset;
                    float twiddleReal = (float) Math.cos(angle);
                    float twiddleImag = (float) Math.sin(angle);
                    int evenIndex = start + offset;
                    int oddIndex = evenIndex + halfSize;

                    float oddReal = real[oddIndex] * twiddleReal - imag[oddIndex] * twiddleImag;
                    float oddImag = real[oddIndex] * twiddleImag + imag[oddIndex] * twiddleReal;

                    real[oddIndex] = real[evenIndex] - oddReal;
                    imag[oddIndex] = imag[evenIndex] - oddImag;
                    real[evenIndex] += oddReal;
                    imag[evenIndex] += oddImag;
                }
            }
        }
    }

    private static float[] readFloatArray(JSONArray array) throws JSONException {
        float[] values = new float[array.length()];
        for (int index = 0; index < array.length(); index++) {
            values[index] = (float) array.getDouble(index);
        }
        return values;
    }

    private static String[] readStringArray(JSONArray array) throws JSONException {
        String[] values = new String[array.length()];
        for (int index = 0; index < array.length(); index++) {
            values[index] = array.getString(index);
        }
        return values;
    }

    private static String readAssetText(Context context, String assetName) throws IOException {
        try (InputStream inputStream = context.getAssets().open(assetName)) {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            return outputStream.toString(StandardCharsets.UTF_8.name());
        }
    }

    private static MappedByteBuffer loadModelFile(Context context, String assetName) throws IOException {
        AssetFileDescriptor descriptor = context.getAssets().openFd(assetName);
        try (FileInputStream inputStream = new FileInputStream(descriptor.getFileDescriptor());
             FileChannel fileChannel = inputStream.getChannel()) {
            return fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.getStartOffset(),
                    descriptor.getDeclaredLength()
            );
        }
    }

    public static final class Prediction {
        public final int actionId;
        public final String label;
        public final float confidence;
        public final float[] probabilities;

        Prediction(int actionId, String label, float confidence, float[] probabilities) {
            this.actionId = actionId;
            this.label = label;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }
    }
}
