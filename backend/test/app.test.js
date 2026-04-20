const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");
const { createApp } = require("../src/app");

function makeTempWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "voice-collector-backend-"));
}

async function withServer(testContext, callback) {
  const workspace = makeTempWorkspace();
  const storageRoot = path.join(workspace, "audio");
  const databaseFile = path.join(workspace, "recordings.sqlite");
  const { app, database } = createApp({ storageRoot, databaseFile });
  const server = app.listen(0);

  testContext.after(() => {
    database.close();
    server.close();
    fs.rmSync(workspace, { recursive: true, force: true });
  });

  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;
  return callback(baseUrl, workspace);
}

test("health endpoint reports ok", async (t) => {
  await withServer(t, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`);
    assert.equal(response.status, 200);
    const payload = await response.json();
    assert.equal(payload.ok, true);
    assert.equal(payload.recordingCount, 0);
  });
});

test("upload creates a recording and file endpoint streams it", async (t) => {
  await withServer(t, async (baseUrl) => {
    const body = new FormData();
    body.set("note", "sample note");
    body.set("duration_ms", "1500");
    body.set(
      "audio",
      new File([Buffer.from("fake-audio-data")], "sample.m4a", { type: "audio/mp4" })
    );

    const uploadResponse = await fetch(`${baseUrl}/recordings`, {
      method: "POST",
      body,
    });

    assert.equal(uploadResponse.status, 201);
    const uploadPayload = await uploadResponse.json();
    assert.equal(uploadPayload.recording.note, "sample note");
    assert.equal(uploadPayload.recording.durationMs, 1500);

    const listResponse = await fetch(`${baseUrl}/recordings`);
    const listPayload = await listResponse.json();
    assert.equal(listPayload.recordings.length, 1);

    const fileResponse = await fetch(uploadPayload.recording.fileUrl);
    assert.equal(fileResponse.status, 200);
    assert.equal(await fileResponse.text(), "fake-audio-data");
  });
});

test("upload requires an audio file", async (t) => {
  await withServer(t, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/recordings`, {
      method: "POST",
      body: new FormData(),
    });

    assert.equal(response.status, 400);
    const payload = await response.json();
    assert.match(payload.error, /audio file/i);
  });
});
