const path = require("path");

const ROOT_DIR = path.resolve(__dirname, "..");
const STORAGE_ROOT = process.env.VOICE_STORAGE_DIR || path.join(ROOT_DIR, "storage", "audio");
const DATABASE_FILE =
  process.env.VOICE_DATABASE_FILE || path.join(ROOT_DIR, "storage", "recordings.sqlite");
const PORT = Number(process.env.PORT || 3000);

module.exports = {
  ROOT_DIR,
  STORAGE_ROOT,
  DATABASE_FILE,
  PORT,
};
