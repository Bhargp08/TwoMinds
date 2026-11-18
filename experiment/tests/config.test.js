import { describe, expect, it } from "vitest";
import {
  N_TRIALS_PER_BLOCK,
  NO_PRESSURE_STAGE_MS,
  TIME_PRESSURE_STAGE1_MS,
  TIME_PRESSURE_STAGE2_MS,
  deadlinesForCondition,
  loadConfig,
} from "../src/lib/config.js";

describe("timing config constants", () => {
  it("exposes pilot-tunable deadlines", () => {
    expect(NO_PRESSURE_STAGE_MS).toBe(3000);
    expect(TIME_PRESSURE_STAGE1_MS).toBe(1000);
    expect(TIME_PRESSURE_STAGE2_MS).toBe(1500);
    expect(N_TRIALS_PER_BLOCK).toBe(150);
  });

  it("maps conditions to stage deadlines", () => {
    expect(deadlinesForCondition("no_pressure")).toEqual({
      stage1: NO_PRESSURE_STAGE_MS,
      stage2: NO_PRESSURE_STAGE_MS,
    });
    expect(deadlinesForCondition("time_pressure")).toEqual({
      stage1: TIME_PRESSURE_STAGE1_MS,
      stage2: TIME_PRESSURE_STAGE2_MS,
    });
  });
});

describe("loadConfig", () => {
  it("defaults to mock when no DataPipe ID is set", () => {
    expect(loadConfig({})).toEqual({ mode: "mock", experimentId: null });
  });

  it("uses production when a DataPipe ID is provided", () => {
    expect(loadConfig({ experimentId: "abc123" })).toEqual({
      mode: "production",
      experimentId: "abc123",
    });
  });
});
