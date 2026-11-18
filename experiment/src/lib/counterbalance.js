const CONDITIONS = ["no_pressure", "time_pressure"];

/** Stable hash of participant code → non-negative int. */
export function hashCode(code) {
  let h = 2166136261;
  for (let i = 0; i < code.length; i++) {
    h ^= code.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Counterbalance block order by participant code.
 * Even hash → no_pressure first; odd → time_pressure first.
 */
export function blockOrderForCode(code) {
  if (typeof code !== "string" || code.length === 0) {
    throw new Error("participant code required");
  }
  return hashCode(code) % 2 === 0 ? [...CONDITIONS] : [...CONDITIONS].reverse();
}
