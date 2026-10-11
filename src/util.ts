/** Text utilities: language-aware tokenizer, token abstraction, re-indentation,
 *  semantic shape keys, distribution mixing. Whitespace stays verbatim so
 *  snippets reconstruct exactly. */
import { profileFor } from './languages';

export const KEYWORDS = profileFor(undefined).keywords; // legacy union (compat)

const REGEX_CACHE = new Map<string, RegExp>();

function tokenizerFor(lang?: string): RegExp {
  const p = profileFor(lang);
  const ck = [p.slashComments, p.hashComments, p.dashComments, p.lispComments, p.blockComments, p.tripleQuotes, p.backtickStrings].map(Number).join('');
  let re = REGEX_CACHE.get(ck);
  if (!re) {
    const comments: string[] = [];
    if (p.hashComments) comments.push('(#[^\\n]*)');
    if (p.slashComments) comments.push('(\\/\\/[^\\n]*)');
    if (p.dashComments) comments.push('(--[^\\n]*)');
    if (p.lispComments) comments.push('(;[^\\n]*)');
    if (p.blockComments) comments.push('(\\/\\*[\\s\\S]*?\\*\\/)');
    const commentPart = comments.length ? `(${comments.join('|')})|` : '';
    // '#' prefixes identifiers only in non-#-comment languages (TS private fields).
    const ident = p.hashComments ? '(@[A-Za-z_$][A-Za-z0-9_$]*)' : '([#@]?[A-Za-z_$][A-Za-z0-9_$]*)';
    const triple = p.tripleQuotes ? "('''[\\s\\S]*?'''|\"\"\"[\\s\\S]*?\"\"\")|" : '';
    const backtick = p.backtickStrings ? '|`(?:[^`\\\\]|\\\\.)*`' : '';
    const strings = `("(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*'${backtick})`;
    const ops = '(=>|===|!==|==|!=|<=|>=|&&|\\|\\||\\?\\?|\\?\\.|\\.\\.\\.|\\*\\*|\\+\\+|--|\\+=|-=|\\*=|\\/=|%=|->|::|\\|>)';
    re = new RegExp(`(\\r?\\n)|${commentPart}([ \\t]+)|${ident}|(\\d+(?:\\.\\d+)?)|${triple}${strings}|${ops}|(.)`, 'g');
    REGEX_CACHE.set(ck, re);
  }
  return re;
}

export function tokenize(text: string, lang?: string): string[] {
  const re = tokenizerFor(lang);
  re.lastIndex = 0;
  const tokens: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) tokens.push(m[0]);
  return tokens;
}

/** Abstract token class, parameterized by the language's keyword set. */
export function abstractToken(t: string, lang?: string): string {
  if (t === '\n' || t === '\r\n') return t;
  if (/^[ \t]+$/.test(t)) return t;
  const p = profileFor(lang);
  const isComment =
    (p.slashComments && t.startsWith('//')) ||
    (p.blockComments && t.startsWith('/*')) ||
    (p.hashComments && t.startsWith('#')) ||
    (p.dashComments && t.startsWith('--')) ||
    (p.lispComments && t.startsWith(';'));
  if (isComment) return '#C';
  if (/^["'`]/.test(t)) return '#S';
  if (/^\d/.test(t)) return '#N';
  if (/^#@?[A-Za-z_$]/.test(t)) {
    return p.keywords.has(t) || p.keywords.has(t.replace(/^[#@]/, '')) ? t : '#I';
  }
  return t;
}

export function countNewlines(t: string): number {
  let n = 0;
  for (let i = 0; i < t.length; i++) if (t.charCodeAt(i) === 10) n++;
  return n;
}

export function reindent(text: string, currentIndent: string): string {
  if (!text.includes('\n')) return text;
  const lines = text.split('\n');
  const lead = (s: string) => /^[ \t]*/.exec(s)![0].length;
  let base = Infinity;
  for (let i = 1; i < lines.length; i++) if (lines[i].trim()) base = Math.min(base, lead(lines[i]));
  if (!isFinite(base)) base = 0;
  const delta = currentIndent.length - base;
  if (delta === 0) return text;
  const padChar = currentIndent.length ? currentIndent[currentIndent.length - 1] : ' ';
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) { lines[i] = ''; continue; }
    const l = lead(lines[i]);
    if (delta > 0) lines[i] = padChar.repeat(delta) + lines[i];
    else lines[i] = lines[i].slice(Math.min(-delta, l));
  }
  return lines.join('\n');
}

export function tail(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : text.slice(text.length - maxChars);
}

export function hashText(text: string): number {
  let hash = 5381;
  for (let i = 0; i < text.length; i++) hash = ((hash << 5) + hash + text.charCodeAt(i)) | 0;
  return hash >>> 0;
}

export function mixDistributions(base: Map<string, number>, boost: Map<string, number>, boostD: number): Map<string, number> {
  let mass = 0;
  for (const w of boost.values()) mass += w;
  if (mass <= 0) return base;
  const lambda = mass / (mass + boostD);
  const out = new Map<string, number>();
  for (const [t, p] of base) out.set(t, (1 - lambda) * p);
  for (const [t, w] of boost) out.set(t, (out.get(t) ?? 0) + (lambda * w) / mass);
  return out;
}

/** Semantic action key: identifiers/numbers/strings become index-linked
 *  placeholders (#I1, #I2, ...), whitespace runs collapse to one space,
 *  keywords/operators stay literal. Renamed variants share one key. */
export function shapeKeyOf(lang: string, tokens: string[]): string {
  const seen = new Map<string, string>();
  let next = 1;
  const parts: string[] = [];
  for (const t of tokens) {
    if (t === '\n' || t === '\r\n') { parts.push(t); continue; }
    if (/^[ \t]+$/.test(t)) { parts.push(' '); continue; }
    const a = abstractToken(t, lang);
    if (a === '#I' || a === '#N' || a === '#S') {
      let tag = seen.get(t);
      if (!tag) { tag = a + next++; seen.set(t, tag); }
      parts.push(tag);
    } else parts.push(a);
  }
  return lang + '\u0003' + parts.join('\u0000');
}
