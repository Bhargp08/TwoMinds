// Ambiguity-free alphabet (MindBench convention): A-Z without I,L,O + digits 2-9.
export const CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
export const CODE_PREFIX = "TM-";
export const CODE_BODY_LENGTH = 4;

function defaultBytes(n) {
  const c = globalThis.crypto;
  if (!c || typeof c.getRandomValues !== "function") {
    throw new Error("Secure random source (crypto.getRandomValues) is not available");
  }
  return c.getRandomValues(new Uint8Array(n));
}

/**
 * Random anonymous participant code, e.g. TM-7F3K.
 * No names, emails, or IPs — rejection sampling over CODE_ALPHABET.
 */
export function generateParticipantCode(getBytes = defaultBytes) {
  const limit = 256 - (256 % CODE_ALPHABET.length);
  let body = "";
  while (body.length < CODE_BODY_LENGTH) {
    for (const b of getBytes(CODE_BODY_LENGTH * 2)) {
      if (b < limit) body += CODE_ALPHABET[b % CODE_ALPHABET.length];
      if (body.length === CODE_BODY_LENGTH) break;
    }
  }
  return CODE_PREFIX + body;
}

export function isValidParticipantCode(code) {
  if (typeof code !== "string" || !code.startsWith(CODE_PREFIX)) return false;
  const body = code.slice(CODE_PREFIX.length);
  return (
    body.length === CODE_BODY_LENGTH &&
    [...body].every((ch) => CODE_ALPHABET.includes(ch))
  );
}
