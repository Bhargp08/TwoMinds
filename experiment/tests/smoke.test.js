import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { runSmokeSession } from "../scripts/smoke.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");

describe("headless smoke session (mock-save)", () => {
  it("writes a CSV with expected columns and short trial counts", async () => {
    const outPath = join(ROOT, "tests/fixtures/smoke_session.csv");
    const { participantCode, rows, path } = await runSmokeSession({
      outPath,
      nTrials: 2,
      seed: 42,
    });
    expect(isValidLike(participantCode)).toBe(true);
    expect(path).toBe(outPath);
    expect(rows.length).toBe(4); // 2 blocks × 2 trials
    const text = readFileSync(outPath, "utf8");
    const header = text.trim().split("\n")[0];
    for (const col of [
      "participant_code",
      "block",
      "condition",
      "trial",
      "a1",
      "s2",
      "a2",
      "r",
      "rt1",
      "rt2",
      "transition",
      "missed_a1",
      "missed_a2",
      "p_s1_a0",
      "p_s1_a1",
      "p_s2_a0",
      "p_s2_a1",
      "schedule_seed",
    ]) {
      expect(header).toContain(col);
    }
  });
});

function isValidLike(code) {
  return typeof code === "string" && /^TM-[A-Z2-9]{4}$/.test(code);
}
