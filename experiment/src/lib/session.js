import {
  SCHEDULE_SEED_NO_PRESSURE,
  SCHEDULE_SEED_TIME_PRESSURE,
  deadlinesForCondition,
} from "./config.js";
import { blockOrderForCode } from "./counterbalance.js";
import { getRewardSchedule } from "./schedule.js";

const P_COMMON = 0.7;

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function seedFromCode(code) {
  let h = 2166136261;
  for (let i = 0; i < code.length; i++) {
    h ^= code.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Simulate one block of the two-step task (used by headless smoke).
 * Choices are random; transitions/rewards use a seeded RNG so smoke is deterministic
 * when `rngSeed` is fixed. Missed trials are not injected here.
 */
export function simulateBlock({
  participantCode,
  block,
  condition,
  nTrials,
  scheduleSeed,
  rngSeed,
  missRate = 0,
}) {
  const schedule = getRewardSchedule(scheduleSeed);
  if (nTrials > schedule.length) {
    throw new Error(`nTrials ${nTrials} exceeds schedule length ${schedule.length}`);
  }
  const rng = mulberry32(rngSeed ?? seedFromCode(participantCode) + block * 997);
  const deadlines = deadlinesForCondition(condition);
  const rows = [];
  for (let trial = 0; trial < nTrials; trial++) {
    const miss1 = rng() < missRate;
    let a1 = null;
    let s2 = null;
    let a2 = null;
    let r = null;
    let transition = null;
    let rt1 = null;
    let rt2 = null;
    let missed_a1 = false;
    let missed_a2 = false;

    if (miss1) {
      missed_a1 = true;
      rt1 = deadlines.stage1;
    } else {
      a1 = rng() < 0.5 ? 0 : 1;
      rt1 = Math.floor(rng() * deadlines.stage1 * 0.8) + 50;
      const common = rng() < P_COMMON;
      const commonState = 1 + a1;
      s2 = common ? commonState : 3 - commonState;
      transition = common ? "common" : "rare";
      const miss2 = rng() < missRate;
      if (miss2) {
        missed_a2 = true;
        rt2 = deadlines.stage2;
      } else {
        a2 = rng() < 0.5 ? 0 : 1;
        rt2 = Math.floor(rng() * deadlines.stage2 * 0.8) + 50;
        const p = schedule[trial][s2 - 1][a2];
        r = rng() < p ? 1 : 0;
      }
    }

    const probs = schedule[trial];
    rows.push({
      participant_code: participantCode,
      block,
      condition,
      trial,
      a1: a1 == null ? "" : a1,
      s2: s2 == null ? "" : s2,
      a2: a2 == null ? "" : a2,
      r: r == null ? "" : r,
      rt1: rt1 == null ? "" : rt1,
      rt2: rt2 == null ? "" : rt2,
      transition: transition ?? "",
      missed_a1,
      missed_a2,
      p_s1_a0: probs[0][0],
      p_s1_a1: probs[0][1],
      p_s2_a0: probs[1][0],
      p_s2_a1: probs[1][1],
      schedule_seed: scheduleSeed,
    });
  }
  return rows;
}

export function scheduleSeedForCondition(condition) {
  if (condition === "no_pressure") return SCHEDULE_SEED_NO_PRESSURE;
  if (condition === "time_pressure") return SCHEDULE_SEED_TIME_PRESSURE;
  throw new Error(`Unknown condition "${condition}"`);
}

/** Run both blocks for a participant (order counterbalanced by code). */
export function runSessionRows({ participantCode, nTrials, rngSeed, missRate = 0 }) {
  const order = blockOrderForCode(participantCode);
  const rows = [];
  order.forEach((condition, block) => {
    rows.push(
      ...simulateBlock({
        participantCode,
        block,
        condition,
        nTrials,
        scheduleSeed: scheduleSeedForCondition(condition),
        rngSeed: (rngSeed ?? 0) + block * 10_000,
        missRate,
      }),
    );
  });
  return { order, rows };
}
