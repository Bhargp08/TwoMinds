#!/usr/bin/env node
/**
 * Headless smoke run of a short session in mock-save mode.
 * Writes tests/fixtures/smoke_session.csv for TM-07.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { generateParticipantCode } from "../src/lib/participantCode.js";
import { createSaver, trialsToCsv } from "../src/lib/save.js";
import { runSessionRows } from "../src/lib/session.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");

export async function runSmokeSession({
  outPath = join(ROOT, "tests/fixtures/smoke_session.csv"),
  nTrials = 2,
  seed = 42,
  participantCode,
} = {}) {
  // Deterministic code for fixture stability when seed is fixed.
  const code =
    participantCode ??
    generateParticipantCode(() => {
      const bytes = new Uint8Array(8);
      let x = seed >>> 0;
      for (let i = 0; i < bytes.length; i++) {
        x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
        bytes[i] = x & 0xff;
      }
      return bytes;
    });

  const { rows } = runSessionRows({
    participantCode: code,
    nTrials,
    rngSeed: seed,
  });

  const csv = trialsToCsv(rows);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, csv);

  const saver = createSaver({ mode: "mock" });
  await saver.save(`${code}_session.csv`, csv);

  return { participantCode: code, rows, path: outPath, saver };
}

const invokedDirectly =
  Boolean(process.argv[1]) &&
  fileURLToPath(import.meta.url) === resolve(process.argv[1]);

if (invokedDirectly) {
  const nTrials = Number(process.env.SMOKE_N_TRIALS ?? 2);
  const result = await runSmokeSession({ nTrials });
  console.log(
    `smoke ok: ${result.rows.length} rows → ${result.path} (${result.participantCode})`,
  );
}
