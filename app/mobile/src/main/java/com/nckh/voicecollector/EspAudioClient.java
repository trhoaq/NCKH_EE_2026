package com.nckh.voicecollector;

import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class EspAudioClient implements Closeable {
    private static final int MAGIC = 0x56435031;
    private static final short TYPE_HELLO = 1;
    private static final short TYPE_SESSION_START = 2;
    private static final short TYPE_AUDIO_FRAME = 3;
    private static final short TYPE_SESSION_END = 4;
    private static final short TYPE_STATE = 5;
    private static final short TYPE_DETECTION = 6;
    private static final short TYPE_ERROR = 7;

    private static final int CONNECT_TIMEOUT_MS = 5000;
    private static final int SO_TIMEOUT_MS = 300;

    private final String host;
    private final int port;

    private Socket socket;
    private DataInputStream inputStream;
    private DataOutputStream outputStream;

    public EspAudioClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public ServerHello connectAndReadHello() throws IOException {
        socket = new Socket();
        socket.connect(new InetSocketAddress(host, port), CONNECT_TIMEOUT_MS);
        socket.setTcpNoDelay(true);
        socket.setSoTimeout(SO_TIMEOUT_MS);

        inputStream = new DataInputStream(socket.getInputStream());
        outputStream = new DataOutputStream(socket.getOutputStream());

        PacketHeader header = readHeader();
        if (header.type != TYPE_HELLO) {
            throw new IOException("Expected HELLO packet from ESP");
        }
        if (header.length != 24) {
            throw new IOException("Unexpected HELLO payload size: " + header.length);
        }

        return new ServerHello(
                inputStream.readInt(),
                inputStream.readInt(),
                inputStream.readInt(),
                inputStream.readUnsignedShort(),
                inputStream.readUnsignedShort(),
                inputStream.readInt(),
                inputStream.readInt() == 1
        );
    }

    public void sendSessionStart(int sessionId, ServerHello hello, int keywordSetVersion) throws IOException {
        writeHeader(TYPE_SESSION_START, 20);
        outputStream.writeInt(sessionId);
        outputStream.writeInt(hello.sampleRate);
        outputStream.writeInt(hello.frameSamples);
        outputStream.writeShort(hello.bitsPerSample);
        outputStream.writeShort(hello.channels);
        outputStream.writeInt(keywordSetVersion);
        outputStream.flush();
    }

    public void sendAudioFrame(int sessionId, int sequence, short[] samples, int sampleCount) throws IOException {
        byte[] pcmBytes = encodePcm16Le(samples, sampleCount);
        writeHeader(TYPE_AUDIO_FRAME, 8 + pcmBytes.length);
        outputStream.writeInt(sessionId);
        outputStream.writeInt(sequence);
        outputStream.write(pcmBytes);
        outputStream.flush();
    }

    public void sendSessionEnd(int sessionId, int totalFrames) throws IOException {
        writeHeader(TYPE_SESSION_END, 8);
        outputStream.writeInt(sessionId);
        outputStream.writeInt(totalFrames);
        outputStream.flush();
    }

    public PollResult drainResponses() throws IOException {
        PollResult result = new PollResult();

        while (true) {
            try {
                PacketHeader header = readHeader();
                switch (header.type) {
                    case TYPE_STATE:
                        readStatePayload(result, header.length);
                        break;
                    case TYPE_DETECTION:
                        readDetectionPayload(result, header.length);
                        break;
                    case TYPE_ERROR:
                        readErrorPayload(result, header.length);
                        break;
                    default:
                        skipPayload(header.length);
                        break;
                }
            } catch (SocketTimeoutException timeoutException) {
                break;
            }
        }

        return result;
    }

    private void readStatePayload(PollResult result, int payloadLength) throws IOException {
        if (payloadLength != 8) {
            skipPayload(payloadLength);
            result.lastError = "Malformed STATE packet from ESP";
            return;
        }
        int sessionId = inputStream.readInt();
        int stateCode = inputStream.readInt();
        result.lastState = new ServerState(sessionId, stateCode, mapStateCode(stateCode));
    }

    private void readDetectionPayload(PollResult result, int payloadLength) throws IOException {
        if (payloadLength < 12) {
            skipPayload(payloadLength);
            result.lastError = "Malformed DETECTION packet from ESP";
            return;
        }

        int sessionId = inputStream.readInt();
        int actionId = inputStream.readUnsignedShort();
        int keywordLength = inputStream.readUnsignedShort();
        int scoreMilli = inputStream.readInt();
        int remainingBytes = payloadLength - 12;
        byte[] keywordBytes = new byte[Math.min(keywordLength, Math.max(remainingBytes, 0))];
        if (keywordBytes.length > 0) {
            inputStream.readFully(keywordBytes);
        }
        if (remainingBytes > keywordBytes.length) {
            skipPayload(remainingBytes - keywordBytes.length);
        }
        String keyword = new String(keywordBytes, StandardCharsets.UTF_8);
        result.detections.add(new Detection(
                sessionId,
                actionId,
                keyword,
                scoreMilli / 1000f,
                mapActionLabel(actionId, keyword)
        ));
    }

    private void readErrorPayload(PollResult result, int payloadLength) throws IOException {
        if (payloadLength < 8) {
            skipPayload(payloadLength);
            result.lastError = "Malformed ERROR packet from ESP";
            return;
        }

        int sessionId = inputStream.readInt();
        int errorCode = inputStream.readUnsignedShort();
        int messageLength = inputStream.readUnsignedShort();
        int remainingBytes = payloadLength - 8;
        byte[] messageBytes = new byte[Math.min(messageLength, Math.max(remainingBytes, 0))];
        if (messageBytes.length > 0) {
            inputStream.readFully(messageBytes);
        }
        if (remainingBytes > messageBytes.length) {
            skipPayload(remainingBytes - messageBytes.length);
        }
        result.lastError = "ESP error " + errorCode + " (session " + sessionId + "): "
                + new String(messageBytes, StandardCharsets.UTF_8);
    }

    private void writeHeader(short type, int payloadLength) throws IOException {
        outputStream.writeInt(MAGIC);
        outputStream.writeShort(type);
        outputStream.writeShort(0);
        outputStream.writeInt(payloadLength);
    }

    private PacketHeader readHeader() throws IOException {
        int magic;
        try {
            magic = inputStream.readInt();
        } catch (EOFException eofException) {
            throw new IOException("Connection closed by ESP", eofException);
        }
        if (magic != MAGIC) {
            throw new IOException("Invalid packet magic from ESP");
        }
        short type = inputStream.readShort();
        inputStream.readShort();
        int length = inputStream.readInt();
        return new PacketHeader(type, length);
    }

    private void skipPayload(int payloadLength) throws IOException {
        int remainingBytes = payloadLength;
        while (remainingBytes > 0) {
            int skipped = inputStream.skipBytes(remainingBytes);
            if (skipped <= 0) {
                break;
            }
            remainingBytes -= skipped;
        }
    }

    private byte[] encodePcm16Le(short[] samples, int sampleCount) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(sampleCount * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int index = 0; index < sampleCount; index++) {
            byteBuffer.putShort(samples[index]);
        }
        return byteBuffer.array();
    }

    private String mapStateCode(int stateCode) {
        switch (stateCode) {
            case 0:
                return "ready";
            case 1:
                return "receiving";
            case 2:
                return "processing";
            case 3:
                return "busy";
            default:
                return "unknown(" + stateCode + ")";
        }
    }

    private String mapActionLabel(int actionId, String keyword) {
        switch (actionId) {
            case 1:
                return "TURN_ON";
            case 2:
                return "TURN_OFF";
            default:
                return keyword == null || keyword.isEmpty() ? "UNKNOWN" : keyword;
        }
    }

    @Override
    public void close() throws IOException {
        IOException closeError = null;

        if (inputStream != null) {
            try {
                inputStream.close();
            } catch (IOException ioException) {
                closeError = ioException;
            }
        }
        if (outputStream != null) {
            try {
                outputStream.close();
            } catch (IOException ioException) {
                closeError = ioException;
            }
        }
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException ioException) {
                closeError = ioException;
            }
        }

        if (closeError != null) {
            throw closeError;
        }
    }

    private static final class PacketHeader {
        final short type;
        final int length;

        PacketHeader(short type, int length) {
            this.type = type;
            this.length = length;
        }
    }

    public static final class ServerHello {
        public final int protocolVersion;
        public final int sampleRate;
        public final int frameSamples;
        public final int bitsPerSample;
        public final int channels;
        public final int wakewordCount;
        public final boolean detectorReady;

        ServerHello(
                int protocolVersion,
                int sampleRate,
                int frameSamples,
                int bitsPerSample,
                int channels,
                int wakewordCount,
                boolean detectorReady
        ) {
            this.protocolVersion = protocolVersion;
            this.sampleRate = sampleRate;
            this.frameSamples = frameSamples;
            this.bitsPerSample = bitsPerSample;
            this.channels = channels;
            this.wakewordCount = wakewordCount;
            this.detectorReady = detectorReady;
        }
    }

    public static final class ServerState {
        public final int sessionId;
        public final int stateCode;
        public final String stateLabel;

        ServerState(int sessionId, int stateCode, String stateLabel) {
            this.sessionId = sessionId;
            this.stateCode = stateCode;
            this.stateLabel = stateLabel;
        }
    }

    public static final class Detection {
        public final int sessionId;
        public final int actionId;
        public final String keyword;
        public final float score;
        public final String actionLabel;

        Detection(int sessionId, int actionId, String keyword, float score, String actionLabel) {
            this.sessionId = sessionId;
            this.actionId = actionId;
            this.keyword = keyword;
            this.score = score;
            this.actionLabel = actionLabel;
        }
    }

    public static final class PollResult {
        public final List<Detection> detections = new ArrayList<>();
        public ServerState lastState;
        public String lastError;
    }
}
