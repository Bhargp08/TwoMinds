import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { getRewardSchedule } from "../src/lib/schedule.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const FIXTURE_DIR = join(ROOT, "tests/fixtures");

function loadFixtures() {
  return readdirSync(FIXTURE_DIR)
    .filter((name) => /^schedule_seed\d+\.json$/.test(name))
    .map((name) => JSON.parse(readFileSync(join(FIXTURE_DIR, name), "utf8")));
}

describe("getRewardSchedule (shipped numpy schedules)", () => {
  it("matches each Python fixture to 1e-9 for the same seed", () => {
    const fixtures = loadFixtures();
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const fixture of fixtures) {
      const got = getRewardSchedule(fixture.seed);
      expect(got.length).toBe(fixture.n_trials);
      for (let t = 0; t < fixture.n_trials; t++) {
        for (let s = 0; s < 2; s++) {
          for (let a = 0; a < 2; a++) {
            expect(Math.abs(got[t][s][a] - fixture.schedule[t][s][a])).toBeLessThanOrEqual(1e-9);
          }
        }
      }
    }
  });

  it("throws for an unknown seed", () => {
    expect(() => getRewardSchedule(999999)).toThrow(/unknown seed/i);
  });
});
