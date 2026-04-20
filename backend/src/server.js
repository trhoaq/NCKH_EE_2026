const { createApp } = require("./app");
const { DATABASE_FILE, PORT, STORAGE_ROOT } = require("./config");

const { app } = createApp({
  storageRoot: STORAGE_ROOT,
  databaseFile: DATABASE_FILE,
});

app.listen(PORT, () => {
  console.log(`Voice collector backend listening on http://localhost:${PORT}`);
});
