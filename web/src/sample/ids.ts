// kta:fixture sample data before the store exists (D95); deleted in WP9a phase B.
// Deterministic look-alike identifiers so the sample is stable between reloads.

function fnv(seed: string, salt: number): number {
  let h = 0x811c9dc5 ^ salt;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** 64-hex record id, shaped like a content hash. */
export function rid(label: string): string {
  let out = "";
  for (let i = 0; out.length < 64; i++) out += fnv(label, i).toString(16).padStart(8, "0");
  return out.slice(0, 64);
}

const CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/** 26-char ULID-shaped runtime id (loops, tasks, jobs, spans). */
export function ulid(label: string): string {
  let out = "01JB";
  for (let i = 0; out.length < 26; i++) {
    const h = fnv(label, i + 101);
    out += CROCKFORD[h % 32] + CROCKFORD[(h >>> 5) % 32];
  }
  return out.slice(0, 26);
}

/** The sample's fixed "now". */
export const AS_OF = "2026-10-30T14:30:00Z";
const AS_OF_MS = Date.parse(AS_OF);

/** ISO timestamp `min` minutes before the sample's now. */
export function ago(min: number): string {
  return new Date(AS_OF_MS - Math.round(min * 60_000)).toISOString().replace(".000Z", "Z");
}

/** ISO timestamp at hh:mm(:ss) on the sample day (UTC). */
export function at(hhmm: string): string {
  const [h, m, s] = hhmm.split(":").map(Number);
  return new Date(Date.UTC(2026, 9, 30, h, m, s || 0)).toISOString().replace(".000Z", "Z");
}
