import { describe, expect, it } from "vitest";
import { blockOrderForCode } from "../src/lib/counterbalance.js";

describe("blockOrderForCode", () => {
  it("returns the two conditions in some order", () => {
    const order = blockOrderForCode("TM-7F3K");
    expect(order).toHaveLength(2);
    expect(new Set(order)).toEqual(new Set(["no_pressure", "time_pressure"]));
  });

  it("is deterministic for a given code", () => {
    expect(blockOrderForCode("TM-AAAA")).toEqual(blockOrderForCode("TM-AAAA"));
  });

  it("counterbalances: different codes can reverse order", () => {
    const seen = new Set();
    const codes = ["TM-7F3K", "TM-AAAA", "TM-ZZZZ", "TM-2345", "TM-HJKM", "TM-PQRS", "TM-WXYZ", "TM-2ABC"];
    for (const code of codes) {
      seen.add(blockOrderForCode(code).join("|"));
    }
    expect(seen.size).toBe(2);
  });
});
