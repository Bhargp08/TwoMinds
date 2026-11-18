/** Pilot-tunable timing (ms). */
export const NO_PRESSURE_STAGE_MS = 3000;
export const TIME_PRESSURE_STAGE1_MS = 1000;
export const TIME_PRESSURE_STAGE2_MS = 1500;
export const N_TRIALS_PER_BLOCK = 150;

/** Fixed schedule seeds per condition (must exist in schedules.json). */
export const SCHEDULE_SEED_NO_PRESSURE = 101;
export const SCHEDULE_SEED_TIME_PRESSURE = 202;

export const MODES = ["mock", "production"];

export function deadlinesForCondition(condition) {
  if (condition === "no_pressure") {
    return { stage1: NO_PRESSURE_STAGE_MS, stage2: NO_PRESSURE_STAGE_MS };
  }
  if (condition === "time_pressure") {
    return { stage1: TIME_PRESSURE_STAGE1_MS, stage2: TIME_PRESSURE_STAGE2_MS };
  }
  throw new Error(`Unknown condition "${condition}"`);
}

/**
 * Resolve runtime config. Mock-save when no DataPipe experiment ID is set
 * (MindBench convention).
 */
export function loadConfig({ mode, experimentId } = {}) {
  const id =
    typeof experimentId === "string" && experimentId.trim() !== ""
      ? experimentId.trim()
      : null;
  if (mode == null) {
    return id ? { mode: "production", experimentId: id } : { mode: "mock", experimentId: null };
  }
  if (!MODES.includes(mode)) {
    throw new Error(`Unknown mode "${mode}"; expected one of ${MODES.join(", ")}`);
  }
  if (mode === "production") {
    if (!id) throw new Error("Production mode requires a DataPipe experiment ID");
    return { mode, experimentId: id };
  }
  return { mode: "mock", experimentId: null };
}
