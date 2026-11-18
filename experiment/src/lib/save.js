import { buildPipeExtension } from "./pipeExtension.js";

/**
 * Saver factory (MindBench convention).
 * Mock mode keeps copies in memory / writes CSV locally.
 * Production returns the DataPipe extension.
 */
export function createSaver({ mode, experimentId, getFilename, simulate, onSave } = {}) {
  if (mode === "mock") {
    const saved = [];
    return {
      saved,
      async save(filename, data) {
        const copy = typeof data === "string" ? data : JSON.parse(JSON.stringify(data));
        saved.push({ filename, data: copy });
      },
    };
  }
  if (mode === "production") {
    return {
      extensions: [buildPipeExtension({ experimentId, getFilename, simulate, onSave })],
    };
  }
  throw new Error(`Saver for mode "${mode}" is not implemented yet`);
}

/** Flatten trial rows to CSV text (mock-save / DataPipe-compatible). */
export function trialsToCsv(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const escape = (v) => {
    if (v == null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  for (const row of rows) {
    lines.push(cols.map((c) => escape(row[c])).join(","));
  }
  return lines.join("\n") + "\n";
}
