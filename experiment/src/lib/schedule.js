/**
 * Reward schedules are shipped as fixed JSON exported from Python
 * (`twominds.env.make_reward_schedule`). JS does not regenerate them:
 * numpy's Generator is not portable across runtimes.
 */
import schedules from "./schedules.json" with { type: "json" };

export function getRewardSchedule(seed) {
  const key = String(seed);
  const entry = schedules[key];
  if (!entry) {
    throw new Error(`Unknown seed: ${seed}`);
  }
  // Return a deep copy so callers cannot mutate the shipped table.
  return entry.schedule.map((trial) => trial.map((row) => row.slice()));
}

export function knownScheduleSeeds() {
  return Object.keys(schedules).map(Number).sort((a, b) => a - b);
}
