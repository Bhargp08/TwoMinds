import { describe, expect, it } from "vitest";
import {
  CODE_ALPHABET,
  generateParticipantCode,
  isValidParticipantCode,
} from "../src/lib/participantCode.js";

const AMBIGUOUS = ["0", "O", "1", "I", "L"];

describe("generateParticipantCode", () => {
  it("alphabet has no ambiguous characters", () => {
    for (const ch of AMBIGUOUS) expect(CODE_ALPHABET).not.toContain(ch);
    expect(new Set(CODE_ALPHABET).size).toBe(CODE_ALPHABET.length);
  });

  it("returns TM-XXXX from the alphabet", () => {
    const code = generateParticipantCode();
    expect(code).toMatch(/^TM-[A-Z2-9]{4}$/);
    expect(isValidParticipantCode(code)).toBe(true);
    for (const ch of code.slice(3)) expect(CODE_ALPHABET).toContain(ch);
  });

  it("is deterministic for an injected byte source", () => {
    const bytes = () => Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(generateParticipantCode(bytes)).toBe(generateParticipantCode(bytes));
  });

  it("rejects names/emails-shaped strings", () => {
    expect(isValidParticipantCode("Bharg")).toBe(false);
    expect(isValidParticipantCode("user@example.com")).toBe(false);
    expect(isValidParticipantCode("TM-7F3K")).toBe(true);
  });
});
