const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const multer = require("multer");

function buildRelativeStoragePath(now, extension) {
  const year = String(now.getFullYear());
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  const filename = `${crypto.randomUUID()}${extension}`;
  return path.join(year, month, day, filename);
}

function buildUploader(storageRoot) {
  fs.mkdirSync(storageRoot, { recursive: true });

  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      const now = new Date();
      const extension = path.extname(file.originalname || "").toLowerCase() || ".m4a";
      const relativePath = buildRelativeStoragePath(now, extension);
      const absolutePath = path.join(storageRoot, relativePath);

      req.savedRelativePath = relativePath;
      req.savedAbsolutePath = absolutePath;
      req.savedFilename = path.basename(absolutePath);

      fs.mkdirSync(path.dirname(absolutePath), { recursive: true });
      cb(null, path.dirname(absolutePath));
    },
    filename: (req, file, cb) => {
      cb(null, req.savedFilename);
    },
  });

  return multer({
    storage,
    limits: {
      fileSize: 50 * 1024 * 1024,
    },
  });
}

module.exports = {
  buildUploader,
};
